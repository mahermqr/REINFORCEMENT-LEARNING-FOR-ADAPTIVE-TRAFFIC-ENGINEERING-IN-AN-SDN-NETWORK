import sys
import os
import numpy as np

# Add agent and controller to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'agent'))
sys.path.append(os.path.dirname(__file__))
from ddpg_security import DDPGSecurityAgent
from ryu.lib.packet import ethernet, ipv4, ether_types

class SecurityModule:
    """
    Adaptive DDoS Detection & Smart Mitigation Module powered by DDPG Continuous Control.
    Monitors live traffic Shannon entropy, packet arrival rates, and flow statistics.
    Installs high-priority dynamic hardware drop rules upon detecting malicious flooding.
    """
    def __init__(self, controller, state_manager):
        self.controller = controller
        self.state_manager = state_manager
        self.logger = controller.logger
        
        # Initialize DDPG Continuous Control Agent (5 state features, 1 continuous action)
        self.agent = DDPGSecurityAgent(state_size=5, action_size=1, actor_lr=0.001, critic_lr=0.002)
        
        self.prev_state = None
        self.prev_action = None
        self.blocked_ips = set()
        self.metered_ips = {}
        
        # Auto-load trained checkpoint if available
        ckpt_path = os.path.join(os.path.dirname(__file__), '..', 'models', 'ddpg_security.pth')
        if os.path.exists(ckpt_path):
            self.agent.load(ckpt_path)
            self.logger.info("[SecurityModule] Loaded pre-trained DDPG Security checkpoint from %s", ckpt_path)
        
        self.logger.info("[SecurityModule] Initialized with DDPG Continuous Agent on device: %s", self.agent.device)

    def analyze_packet(self, pkt, datapath):
        """
        Analyzes incoming packet against current network telemetry.
        Returns:
            True: Packet is safe (allow to proceed).
            False: Packet is part of a DDoS attack (drop and mitigate).
        """
        ip_pkt = pkt.get_protocol(ipv4.ipv4)
        src_ip = ip_pkt.src if ip_pkt else None

        # Check if source is already blocked
        if src_ip and src_ip in self.blocked_ips:
            return False

        # Get continuous state vector from StateManager
        state = self.state_manager.get_security_state()
        # In live operation, use deterministic policy (add_noise=False)
        action = self.agent.act(state, add_noise=False)

        entropy_feat = state[1]
        pps_feat = state[0]
        # Robust attack condition: significant packet volume and compressed entropy
        actual_attack_condition = (entropy_feat < 0.45 and pps_feat > 0.4) or (pps_feat > 0.75)
        
        # Action indicates mitigation when above decision threshold
        is_attack = (action >= 0.2) and actual_attack_condition

        if is_attack and src_ip:
            self.logger.warning("[SecurityModule] DDoS Attack Detected! Action: %.3f | PPS: %.2f | Entropy: %.3f | Attacker: %s",
                                action, pps_feat * 5000.0, entropy_feat, src_ip)
            self._mitigate_attack(datapath, src_ip)
            return False

        return True

    def check_active_flows(self):
        """Checks telemetry across active flows periodically and mitigates attacks."""
        state = self.state_manager.get_security_state()
        action = self.agent.act(state, add_noise=False)

        pps_feat = state[0]
        entropy_feat = state[1]

        # Check for attack: DDPG action >= 0.0 and low entropy (or severe entropy collapse below 0.35)
        # Legitimate high-rate load-balancing flows have high entropy (~0.90 - 1.0) and normal MTU packets
        is_attack = (action >= 0.0 and entropy_feat < 0.50) or (entropy_feat < 0.35 and pps_feat > 0.2)

        if is_attack:
            offender_ip = None
            max_pkts = 0
            for (dpid, src_ip, dst_ip), stat in self.state_manager.flow_stats.items():
                if stat.get('pps', 0) > max_pkts and stat.get('pps', 0) > 2000:
                    max_pkts = stat.get('pps', 0)
                    offender_ip = src_ip
            
            if not offender_ip and self.state_manager.src_ip_counter and entropy_feat < 0.35:
                offender_ip = max(self.state_manager.src_ip_counter, key=self.state_manager.src_ip_counter.get)

            if offender_ip and offender_ip not in self.blocked_ips:
                severity = "critical" if action >= 0.5 else "moderate"
                self.logger.warning("[SecurityModule] DDoS Attack Detected via Telemetry! Action: %.3f (Tier: %s) | PPS: %.1f | Entropy: %.3f | Attacker: %s",
                                    action, severity, pps_feat * 5000.0, entropy_feat, offender_ip)
                for dp in list(self.controller.datapaths.values()):
                    self._mitigate_attack(dp, offender_ip, severity=severity)

    def _apply_meter_rate_limit(self, datapath, attacker_ip, rate_kbps=5000, meter_id=10):
        """Installs OpenFlow 1.3 Meter to rate-limit suspicious flow without full packet dropping."""
        self.metered_ips[attacker_ip] = rate_kbps
        parser = datapath.ofproto_parser
        ofproto = datapath.ofproto

        bands = [parser.OFPMeterBandDrop(rate=int(rate_kbps), burst_size=int(rate_kbps * 0.2))]
        try:
            del_mod = parser.OFPMeterMod(datapath, ofproto.OFPMC_DELETE, meter_id=meter_id)
            datapath.send_msg(del_mod)
        except Exception:
            pass

        try:
            meter_mod = parser.OFPMeterMod(datapath, ofproto.OFPMC_ADD, ofproto.OFPMF_KBPS,
                                          meter_id=meter_id, bands=bands)
            datapath.send_msg(meter_mod)
        except Exception as e:
            self.logger.debug("[SecurityModule] MeterMod not supported on datapath: %s", e)

        match = parser.OFPMatch(eth_type=ether_types.ETH_TYPE_IP, ipv4_src=attacker_ip)
        inst = [parser.OFPInstructionMeter(meter_id=meter_id)]
        try:
            mod = parser.OFPFlowMod(datapath=datapath, priority=90, match=match,
                                    instructions=inst, idle_timeout=30, hard_timeout=60)
            datapath.send_msg(mod)
            self.logger.info("[SecurityModule] Rate-limited IP %s to %d kbps via Meter %d", attacker_ip, rate_kbps, meter_id)
        except Exception as e:
            self.logger.debug("[SecurityModule] FlowMod meter instruction error: %s", e)

    def _mitigate_attack(self, datapath, attacker_ip, severity="critical"):
        """
        Installs high-priority OpenFlow enforcement:
        - Critical (action >= 0.5): Hardware DROP rule (priority 100)
        - Moderate (0.0 <= action < 0.5): Hardware Rate-Limiting Meter (priority 90)
        """
        if severity == "moderate":
            self._apply_meter_rate_limit(datapath, attacker_ip, rate_kbps=5000)
            return

        self.blocked_ips.add(attacker_ip)
        parser = datapath.ofproto_parser
        match = parser.OFPMatch(eth_type=ether_types.ETH_TYPE_IP, ipv4_src=attacker_ip)
        actions = [] # Drop
        self.controller.add_flow(datapath, priority=100, match=match, actions=actions,
                                idle_timeout=30, hard_timeout=60)
        self.logger.info("[SecurityModule] Installed hardware DROP rule for attacker IP: %s (Priority 100)", attacker_ip)

    def unblock_all(self):
        """Clears blocked IPs and deletes DROP flow rules from all switches."""
        self.blocked_ips.clear()
        self.metered_ips.clear()
        for dp in list(self.controller.datapaths.values()):
            parser = dp.ofproto_parser
            ofproto = dp.ofproto
            match = parser.OFPMatch(eth_type=ether_types.ETH_TYPE_IP)
            mod = parser.OFPFlowMod(
                datapath=dp, command=ofproto.OFPFC_DELETE,
                out_port=ofproto.OFPP_ANY, out_group=ofproto.OFPG_ANY,
                priority=100, match=match
            )
            dp.send_msg(mod)
        self.logger.info("[SecurityModule] Blocked IPs, metered rules, and hardware DROP rules reset across all switches.")

    def get_security_summary(self):
        """Returns consolidated multi-vector entropy and tiered enforcement status."""
        mv = self.state_manager.calculate_multivector_entropy()
        state = self.state_manager.get_security_state()
        action = float(self.agent.act(state, add_noise=False))
        return {
            'blocked_ips': list(self.blocked_ips),
            'metered_ips': self.metered_ips,
            'latest_ddpg_action': round(action, 3),
            'action_tier': 'critical_drop' if action >= 0.5 else ('meter_limit' if action >= 0.0 else 'benign'),
            'multivector_entropy': mv
        }
