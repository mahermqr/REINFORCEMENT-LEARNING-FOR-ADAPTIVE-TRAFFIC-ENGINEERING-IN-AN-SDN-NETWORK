from ryu.base import app_manager
from ryu.controller import ofp_event
from ryu.controller.handler import CONFIG_DISPATCHER, MAIN_DISPATCHER, DEAD_DISPATCHER
from ryu.controller.handler import set_ev_cls
from ryu.ofproto import ofproto_v1_3
from ryu.lib.packet import packet, ethernet, ipv4, arp, ether_types
from ryu.lib import hub
from ryu.topology import event
from ryu.topology.api import get_switch, get_link

import sys
import os
import time

# Add controller and agent directories to path
sys.path.append(os.path.dirname(__file__))
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'agent'))

from state_manager import StateManager
from routing_module import RoutingModule
from web_dashboard import DashboardServer

class MainController(app_manager.RyuApp):
    """
    Main OpenFlow 1.3 SDN Controller for Adaptive Traffic Engineering (EC499).
    Integrates Deep Q-Network (Double DQN) dynamic routing, real-time OpenFlow telemetry polling,
    intelligent host tracking, and control overhead monitoring.
    """
    OFP_VERSIONS = [ofproto_v1_3.OFP_VERSION]

    def __init__(self, *args, **kwargs):
        super(MainController, self).__init__(*args, **kwargs)
        self.datapaths = {}
        self.state_manager = StateManager()

        # Initialize Adaptive Traffic Engineering Routing Module
        self.routing_module = RoutingModule(self, self.state_manager)

        # Initialize Live Web Dashboard & REST Server (port 8080)
        self.dashboard_server = DashboardServer(self, self.state_manager, port=8080)

        self.logger.info("=" * 65)
        self.logger.info("MainController Initialized: Adaptive SDN Traffic Engineering (EC499)")
        self.logger.info("=" * 65)

        # Start periodic telemetry polling thread (every 3 seconds)
        self.monitor_thread = hub.spawn(self._monitor_loop)

    def _monitor_loop(self):
        """Periodically requests Port and Flow statistics from all connected switches."""
        while True:
            self.state_manager.sample_telemetry_history()
            self.state_manager.step_dynamic_flows()

            # Actively sync links from Ryu topology service
            try:
                for s in get_switch(self, None):
                    self.state_manager.register_switch(s.dp.id)
                for link in get_link(self, None):
                    self.state_manager.update_link(
                        link.src.dpid, link.dst.dpid, link.src.port_no, link.dst.port_no,
                        capacity_mbps=100.0, delay_ms=2.0
                    )
            except Exception:
                pass

            for dp in list(self.datapaths.values()):
                self._request_stats(dp)
                self._send_echo_request(dp)

            hub.sleep(3)

    def _request_stats(self, datapath):
        """Sends OFPFlowStatsRequest and OFPPortStatsRequest, accounting for control overhead."""
        ofproto = datapath.ofproto
        parser = datapath.ofproto_parser

        # Flow stats request
        req_flow = parser.OFPFlowStatsRequest(datapath)
        datapath.send_msg(req_flow)
        self.state_manager.record_stats_request(byte_size=56)

        # Port stats request
        req_port = parser.OFPPortStatsRequest(datapath, 0, ofproto.OFPP_ANY)
        datapath.send_msg(req_port)
        self.state_manager.record_stats_request(byte_size=56)

    def _send_echo_request(self, datapath):
        """Sends EchoRequest with timestamp to measure controller-switch RTT latency and link jitter."""
        parser = datapath.ofproto_parser
        data = f"{time.time():.6f}".encode('ascii')
        echo_req = parser.OFPEchoRequest(datapath, data=data)
        datapath.send_msg(echo_req)
        self.state_manager.record_stats_request(byte_size=32)

    @set_ev_cls(ofp_event.EventOFPEchoReply, [MAIN_DISPATCHER, CONFIG_DISPATCHER])
    def echo_reply_handler(self, ev):
        """Measures RTT from EchoReply and updates link latency and jitter."""
        try:
            sent_time = float(ev.msg.data.decode('ascii'))
            rtt_ms = (time.time() - sent_time) * 1000.0
            dpid = ev.msg.datapath.id
            self.state_manager.record_stats_reply(byte_size=32)
            for neighbor in self.state_manager.graph.neighbors(dpid):
                self.state_manager.update_link_latency(dpid, neighbor, delay_ms=max(0.5, rtt_ms / 2.0))
        except Exception:
            pass

    @set_ev_cls(ofp_event.EventOFPSwitchFeatures, CONFIG_DISPATCHER)
    def switch_features_handler(self, ev):
        """Installs default table-miss entry on switch connection."""
        datapath = ev.msg.datapath
        ofproto = datapath.ofproto
        parser = datapath.ofproto_parser

        # Table-miss flow entry: Priority 0 -> Send to Controller
        match = parser.OFPMatch()
        actions = [parser.OFPActionOutput(ofproto.OFPP_CONTROLLER, ofproto.OFPCML_NO_BUFFER)]
        self.add_flow(datapath, priority=0, match=match, actions=actions)
        self.logger.info("[MainController] Switch connected: DPID %016x (Table-miss flow installed)", datapath.id)

    def add_flow(self, datapath, priority, match, actions, buffer_id=None, idle_timeout=0, hard_timeout=0):
        """Helper to install flow entries on an OpenFlow switch with control overhead accounting."""
        ofproto = datapath.ofproto
        parser = datapath.ofproto_parser

        inst = [parser.OFPInstructionActions(ofproto.OFPIT_APPLY_ACTIONS, actions)]
        if buffer_id:
            mod = parser.OFPFlowMod(datapath=datapath, buffer_id=buffer_id,
                                    priority=priority, match=match,
                                    instructions=inst, idle_timeout=idle_timeout,
                                    hard_timeout=hard_timeout)
        else:
            mod = parser.OFPFlowMod(datapath=datapath, priority=priority,
                                    match=match, instructions=inst,
                                    idle_timeout=idle_timeout, hard_timeout=hard_timeout)
        datapath.send_msg(mod)
        self.state_manager.record_flow_mod(byte_size=72)

    @set_ev_cls(ofp_event.EventOFPStateChange, [MAIN_DISPATCHER, DEAD_DISPATCHER])
    def _state_change_handler(self, ev):
        """Tracks active switches joining or leaving."""
        datapath = ev.datapath
        if ev.state == MAIN_DISPATCHER:
            if datapath.id not in self.datapaths:
                self.datapaths[datapath.id] = datapath
                self.state_manager.register_switch(datapath.id)
                self.logger.info("[MainController] Datapath registered: %016x", datapath.id)
        elif ev.state == DEAD_DISPATCHER:
            if datapath.id in self.datapaths:
                del self.datapaths[datapath.id]
                self.state_manager.unregister_switch(datapath.id)
                self.logger.info("[MainController] Datapath unregistered: %016x", datapath.id)

    @set_ev_cls(ofp_event.EventOFPPacketIn, MAIN_DISPATCHER)
    def _packet_in_handler(self, ev):
        """Central event dispatcher for all incoming packet events."""
        msg = ev.msg
        datapath = msg.datapath
        in_port = msg.match['in_port']

        pkt = packet.Packet(msg.data)
        eth = pkt.get_protocols(ethernet.ethernet)[0]

        # Account for Packet-In control overhead
        self.state_manager.record_packet_in(byte_size=len(msg.data) + 32)

        # Ignore LLDP & IPv6 Router Discovery
        if eth.ethertype in (ether_types.ETH_TYPE_LLDP, 0x88cc, 0x86dd):
            return

        # Handle ARP resolution
        arp_pkt = pkt.get_protocol(arp.arp)
        if arp_pkt:
            self._handle_arp(datapath, in_port, eth, arp_pkt, msg)
            return

        # Unicast Traffic Engineering via DQN Agent
        self.routing_module.handle_unicast(ev, pkt)

    def _handle_arp(self, datapath, in_port, eth, arp_pkt, msg):
        """Intelligent ARP handling to discover host locations and prevent broadcast storms."""
        src_ip = arp_pkt.src_ip
        src_mac = arp_pkt.src_mac
        dst_ip = arp_pkt.dst_ip

        # Learn host location
        self.state_manager.record_host(src_ip, src_mac, datapath.id, in_port)

        # Check if destination host is already known
        dst_dpid, dst_port = self.state_manager.get_host_location(ip=dst_ip)

        parser = datapath.ofproto_parser
        ofproto = datapath.ofproto

        if dst_dpid is not None and dst_port is not None:
            # Destination known: forward directly to destination switch port
            target_dp = self.datapaths.get(dst_dpid, datapath)
            actions = [target_dp.ofproto_parser.OFPActionOutput(dst_port)]
            out = target_dp.ofproto_parser.OFPPacketOut(
                datapath=target_dp, buffer_id=ofproto.OFP_NO_BUFFER,
                in_port=ofproto.OFPP_CONTROLLER, actions=actions, data=msg.data
            )
            target_dp.send_msg(out)
        else:
            # Safe loop-free ARP broadcast:
            # Broadcast ONLY across host-facing access ports on edge switches (never into trunk links)
            for dp in list(self.datapaths.values()):
                dp_trunk_ports = set()
                if self.state_manager.graph.has_node(dp.id):
                    for _, _, edge_data in self.state_manager.graph.out_edges(dp.id, data=True):
                        if 'port' in edge_data:
                            dp_trunk_ports.add(edge_data['port'])

                for p_no in dp.ports:
                    if p_no <= ofproto_v1_3.OFPP_MAX and p_no not in dp_trunk_ports:
                        if dp.id == datapath.id and p_no == in_port:
                            continue
                        dp_parser = dp.ofproto_parser
                        actions = [dp_parser.OFPActionOutput(p_no)]
                        out = dp_parser.OFPPacketOut(
                            datapath=dp, buffer_id=ofproto_v1_3.OFP_NO_BUFFER,
                            in_port=ofproto_v1_3.OFPP_CONTROLLER, actions=actions, data=msg.data
                        )
                        dp.send_msg(out)

    @set_ev_cls(ofp_event.EventOFPFlowStatsReply, MAIN_DISPATCHER)
    def _flow_stats_reply_handler(self, ev):
        """Processes flow statistics reports and records control reply overhead."""
        body = ev.msg.body
        dpid = ev.msg.datapath.id
        self.state_manager.record_stats_reply(byte_size=len(body) * 48 + 16)

        for stat in body:
            match = stat.match
            src_ip = match.get('ipv4_src', '')
            dst_ip = match.get('ipv4_dst', '')
            if src_ip or dst_ip:
                self.state_manager.update_flow_stats(
                    dpid, src_ip, dst_ip, stat.packet_count, stat.byte_count, stat.duration_sec
                )

    @set_ev_cls(ofp_event.EventOFPPortStatsReply, MAIN_DISPATCHER)
    def _port_stats_reply_handler(self, ev):
        """Processes port statistics reports and records control reply overhead."""
        body = ev.msg.body
        dpid = ev.msg.datapath.id
        self.state_manager.record_stats_reply(byte_size=len(body) * 64 + 16)

        for stat in body:
            if stat.port_no <= ofproto_v1_3.OFPP_MAX:
                self.state_manager.update_port_stats(
                    dpid, stat.port_no, stat.rx_bytes, stat.tx_bytes,
                    stat.rx_packets, stat.tx_packets, stat.duration_sec
                )

    @set_ev_cls(event.EventSwitchEnter)
    def switch_enter_handler(self, ev):
        """Discovers switch join events."""
        switch = ev.switch.dp.id
        self.state_manager.register_switch(switch)
        self.logger.info("[Topology] Discovered switch DPID: %016x", switch)

    @set_ev_cls(event.EventLinkAdd)
    def link_add_handler(self, ev):
        """Discovers new link additions."""
        src = ev.link.src.dpid
        dst = ev.link.dst.dpid
        src_port = ev.link.src.port_no
        dst_port = ev.link.dst.port_no
        self.state_manager.update_link(src, dst, src_port, dst_port, capacity_mbps=100.0, delay_ms=2.0)
        self.logger.info("[Topology] Discovered Link: Switch %s (port %s) <-> Switch %s (port %s)",
                         src, src_port, dst, dst_port)
