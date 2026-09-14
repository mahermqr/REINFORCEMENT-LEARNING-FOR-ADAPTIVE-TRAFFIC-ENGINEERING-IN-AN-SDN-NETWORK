import json
import os
import sys
from ryu.lib import hub
from http.server import HTTPServer, SimpleHTTPRequestHandler
from urllib.parse import urlparse, parse_qs

import numpy as np

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if os.path.join(BASE_DIR, 'topology') not in sys.path:
    sys.path.append(os.path.join(BASE_DIR, 'topology'))

from topology_library import get_topology, list_available_topologies, ALL_TOPOLOGY_BUILDERS

def numpy_json_serializer(obj):
    if hasattr(obj, 'tolist'):
        return obj.tolist()
    if isinstance(obj, (np.floating, np.float32, np.float64)):
        return float(obj)
    if isinstance(obj, (np.integer, np.int32, np.int64)):
        return int(obj)
    return str(obj)

class DashboardServer:
    """
    Lightweight Embedded HTTP REST Server & Real-Time Web Dashboard for Ryu.
    Serves live network topology, port/flow metrics, RL agent telemetry,
    and security alerts at http://localhost:8080.
    """
    def __init__(self, controller, state_manager, port=8080):
        self.controller = controller
        self.state_manager = state_manager
        self.port = port
        self.logger = controller.logger
        self.static_dir = os.path.join(os.path.dirname(__file__), 'static')
        os.makedirs(self.static_dir, exist_ok=True)

        # Start server in Ryu green thread
        self.thread = hub.spawn(self._run_server)

    def _run_server(self):
        controller = self.controller
        state_manager = self.state_manager
        static_dir = self.static_dir

        class RequestHandler(SimpleHTTPRequestHandler):
            def __init__(self, *args, **kwargs):
                super().__init__(*args, directory=static_dir, **kwargs)

            def do_GET(self):
                parsed = urlparse(self.path)
                p = parsed.path
                query = parse_qs(parsed.query)

                if p == '/api/topology':
                    topo_id = query.get('topo', [None])[0]
                    self._send_json(self._get_topology_data(topo_id))
                elif p == '/api/topologies':
                    self._send_json(list_available_topologies())
                elif p == '/api/stats':
                    self._send_json(self._get_stats_data())
                elif p == '/api/security':
                    self._send_json(self._get_security_data())
                elif p == '/api/rl_metrics':
                    self._send_json(self._get_rl_data())
                elif p == '/api/history':
                    self._send_json(list(state_manager.telemetry_history))
                elif p == '/api/benchmarks':
                    self._send_json(self._get_benchmark_data())
                elif p == '/api/unblock_all':
                    controller.security_module.unblock_all()
                    self._send_json({"status": "success", "message": "All blocked IPs cleared."})
                elif p == '/api/security_multivector':
                    summary = controller.security_module.get_security_summary()
                    self._send_json(summary)
                elif p == '/api/simulate/link_failure':
                    src = int(query.get('src', [1])[0])
                    dst = int(query.get('dst', [2])[0])
                    if not state_manager.graph.has_edge(src, dst) and state_manager.graph.number_of_edges() > 0:
                        src, dst = list(state_manager.graph.edges())[0]
                    ok = state_manager.inject_link_failure(src, dst)
                    if ok:
                        self._send_json({"status": "success", "message": f"💥 Link ({src} <-> {dst}) severed! Fast Failover engaged.", "failed_link": [src, dst]})
                    else:
                        self._send_json({"status": "error", "message": f"Link ({src} <-> {dst}) not found."})
                elif p == '/api/simulate/restore_links':
                    cnt = state_manager.restore_link()
                    self._send_json({"status": "success", "message": f"✅ Restored {cnt} severed link(s) to nominal state."})
                elif p == '/api/simulate/wcmp':
                    nodes = list(state_manager.graph.nodes())
                    if len(nodes) >= 2:
                        src, dst = nodes[0], nodes[-1]
                        import itertools, networkx as nx
                        try:
                            paths = list(itertools.islice(nx.shortest_simple_paths(state_manager.graph, src, dst), 4))
                        except Exception:
                            paths = [[src, dst]]
                        weighted = controller.routing_module.calculate_wcmp_weights(paths)
                        msg = f"⚖️ WCMP flow splitting across {len(weighted)} paths from s{src} to s{dst}."
                        self._send_json({"status": "success", "message": msg, "wcmp_weights": weighted})
                    else:
                        self._send_json({"status": "error", "message": "Insufficient nodes for WCMP."})
                elif p == '/api/simulate/traffic_burst':
                    self._handle_simulate_traffic_burst()
                    self._send_json({"status": "success", "message": "Unicast flow surge injected across pods."})
                elif p == '/api/simulate/core_jamming':
                    state_manager.inject_core_congestion(utilization=0.96)
                    self._send_json({"status": "success", "message": "Core switches saturated to 96%. Double DQN rerouting active."})
                elif p == '/api/simulate/ddos_attack':
                    self._handle_simulate_ddos()
                    self._send_json({"status": "success", "message": "Volumetric DDoS flood injected from 10.0.0.4. DDPG filter engaged."})
                elif p == '/api/simulate/multicast':
                    self._handle_simulate_multicast()
                    self._send_json({"status": "success", "message": "Multicast group 224.1.1.1 stream active via Dueling DQN Steiner tree."})
                elif p == '/api/simulate/switch_topology':
                    topo_id = query.get('topo', ['tree'])[0]
                    msg = self._handle_switch_topology(topo_id)
                    self._send_json({"status": "success", "message": msg, "current_topo": topo_id})
                elif p == '/api/simulate/reset':
                    state_manager.reset_simulation()
                    controller.security_module.unblock_all()
                    self._send_json({"status": "success", "message": "Network simulation state reset to nominal baseline."})
                elif p.startswith('/api/plots/'):
                    img_name = os.path.basename(p)
                    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
                    img_path = os.path.join(base_dir, 'logs', 'plots', img_name)
                    if os.path.exists(img_path) and img_name.endswith('.png'):
                        self.send_response(200)
                        self.send_header('Content-Type', 'image/png')
                        self.send_header('Access-Control-Allow-Origin', '*')
                        self.end_headers()
                        with open(img_path, 'rb') as f:
                            self.wfile.write(f.read())
                        return
                    else:
                        self.send_response(404)
                        self.end_headers()
                        return
                else:
                    # Serve static HTML/JS files
                    if p == '/' or p == '':
                        self.path = '/index.html'
                    return super().do_GET()

            def do_POST(self):
                # Also support POST for all simulation endpoints
                self.do_GET()

            def _handle_simulate_traffic_burst(self):
                # Ensure hosts exist
                state_manager.record_host("10.0.0.1", "00:00:00:00:00:01", 4, 1)
                state_manager.record_host("10.0.0.2", "00:00:00:00:00:02", 4, 2)
                state_manager.record_host("10.0.0.5", "00:00:00:00:00:05", 6, 1)
                state_manager.record_host("10.0.0.6", "00:00:00:00:00:06", 6, 2)

                # Flow 1: h2 -> h5 (Pod 1 to Pod 2) - Double DQN chooses lateral cross-link s4-s6 or core
                state = state_manager.get_routing_state(4, 6)
                act = controller.routing_module.agent.act(state, explore=False)
                paths = [[4, 2, 1, 3, 6], [4, 6], [4, 2, 5, 7, 3, 6]]
                chosen = paths[act % len(paths)]
                state_manager.inject_traffic_flow("10.0.0.2", "10.0.0.5", 4, 6, mbps=28.5, pps=2300, path=chosen)

                # Flow 2: h3 -> h7
                state_manager.inject_traffic_flow("10.0.0.3", "10.0.0.7", 5, 7, mbps=19.2, pps=1600, path=[5, 7])

            def _handle_simulate_ddos(self):
                state_manager.inject_ddos_flood(attacker_ip="10.0.0.4", target_ip="10.0.0.1", pps=5200, bpp=75)
                controller.security_module.check_active_flows()

            def _handle_simulate_multicast(self):
                state_manager.register_multicast_member("224.1.1.1", 6, 1) # h5
                state_manager.register_multicast_member("224.1.1.1", 7, 1) # h7
                state_manager.register_multicast_member("224.1.1.1", 7, 2) # h8
                state_manager.inject_traffic_flow("10.0.0.2", "224.1.1.1", 4, 6, mbps=20.0, pps=1650, path=[4, 6, 7])

            def _handle_switch_topology(self, topo_id):
                g, meta = get_topology(topo_id)
                state_manager.graph = g.copy()
                state_manager.link_bandwidths.clear()
                state_manager.link_delays.clear()
                state_manager.link_utilization.clear()
                state_manager.link_loss.clear()
                state_manager.server_cpu.clear()
                state_manager.server_ram.clear()
                state_manager.ip_to_location.clear()
                state_manager.host_ip_to_mac.clear()
                state_manager.flow_stats.clear()
                state_manager.raw_flow_stats.clear()
                state_manager.active_simulation_flows.clear()

                for u, v, d in g.edges(data=True):
                    state_manager.update_link(u, v, src_port=1, dst_port=1,
                                             capacity_mbps=d.get('capacity', 100.0),
                                             delay_ms=d.get('delay', 2.0))
                for h in meta.get('hosts', []):
                    state_manager.record_host(h['ip'], h.get('mac', ''), h['switch'], h.get('port', 1))
                return f"Fabric switched to {meta['name']} ({meta['switches']} switches, {g.number_of_edges()//2} links)"

            def _get_benchmark_data(self):
                base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
                res = {}
                eval_file = os.path.join(base_dir, 'logs', 'evaluation_results.json')
                stress_file = os.path.join(base_dir, 'logs', 'stress_test_results.json')
                blind_file = os.path.join(base_dir, 'logs', 'blind_topologies_stress_results.json')
                if os.path.exists(eval_file):
                    with open(eval_file, 'r') as f:
                        res['training'] = json.load(f)
                if os.path.exists(stress_file):
                    with open(stress_file, 'r') as f:
                        res['stress_test'] = json.load(f)
                if os.path.exists(blind_file):
                    with open(blind_file, 'r') as f:
                        res['blind_topologies_stress'] = json.load(f)
                return res

            def _send_json(self, data):
                self.send_response(200)
                self.send_header('Content-Type', 'application/json')
                self.send_header('Access-Control-Allow-Origin', '*')
                self.end_headers()
                self.wfile.write(json.dumps(data, default=numpy_json_serializer).encode('utf-8'))

            def _get_topology_data(self, topo_id=None):
                # If topology links not yet populated by LLDP, initialize baseline 7-switch fabric
                if state_manager.graph.number_of_edges() == 0:
                    g_def, meta_def = get_topology('tree')
                    state_manager.graph = g_def.copy()
                    for u, v, d in g_def.edges(data=True):
                        state_manager.update_link(u, v, src_port=1, dst_port=1, capacity_mbps=d.get('capacity', 100.0), delay_ms=d.get('delay', 2.0))
                    for h in meta_def.get('hosts', []):
                        state_manager.record_host(h['ip'], h.get('mac', ''), h['switch'], h.get('port', 1))

                # If specific topology preview is requested
                if topo_id and topo_id in ALL_TOPOLOGY_BUILDERS:
                    g_req, meta_req = get_topology(topo_id)
                    pos_req = meta_req.get('positions', {})
                    nodes_req = []
                    for n, d in g_req.nodes(data=True):
                        p = pos_req.get(n, {'x': 0.5, 'y': 0.5})
                        nodes_req.append({
                            'id': n,
                            'label': d.get('label', f's{n}'),
                            'type': d.get('type', 'edge'),
                            'x': p['x'],
                            'y': p['y'],
                            'cpu': state_manager.server_cpu.get(n, 0.1),
                            'ram': state_manager.server_ram.get(n, 0.2)
                        })
                    links_req = []
                    for u, v, d in g_req.edges(data=True):
                        if u < v or not g_req.has_edge(v, u):
                            links_req.append({
                                'source': u,
                                'target': v,
                                'utilization': round(d.get('util', 0.05) * 100, 1),
                                'delay': round(d.get('delay', 2.0), 2),
                                'capacity': d.get('capacity', 100.0)
                            })
                    return {'nodes': nodes_req, 'links': links_req, 'hosts': meta_req.get('hosts', []), 'meta': meta_req}

                # Live state_manager topology
                num_nodes = state_manager.graph.number_of_nodes()
                current_meta = {}
                for tid, bld in ALL_TOPOLOGY_BUILDERS.items():
                    tg, tmeta = bld()
                    if tg.number_of_nodes() == num_nodes:
                        current_meta = tmeta
                        break

                pos_map = current_meta.get('positions', {})
                nodes = []
                for n, d in state_manager.graph.nodes(data=True):
                    p = pos_map.get(n, {'x': 0.5, 'y': 0.5})
                    nodes.append({
                        'id': n,
                        'label': d.get('label', f's{n}'),
                        'type': d.get('type', 'edge'),
                        'x': p['x'],
                        'y': p['y'],
                        'cpu': state_manager.server_cpu.get(n, 0.1),
                        'ram': state_manager.server_ram.get(n, 0.2)
                    })

                links = []
                for u, v in state_manager.graph.edges():
                    if u < v or not state_manager.graph.has_edge(v, u):
                        links.append({
                            'source': u,
                            'target': v,
                            'utilization': round(state_manager.link_utilization.get((u, v), 0.0) * 100, 1),
                            'delay': round(state_manager.link_delays.get((u, v), 2.0), 2),
                            'capacity': state_manager.link_bandwidths.get((u, v), 100.0)
                        })

                hosts = []
                for ip, (dpid, port) in state_manager.ip_to_location.items():
                    hosts.append({
                        'ip': ip,
                        'mac': state_manager.host_ip_to_mac.get(ip, ''),
                        'switch': dpid,
                        'port': port
                    })

                return {'nodes': nodes, 'links': links, 'hosts': hosts, 'meta': current_meta}

            def _get_stats_data(self):
                total_mbps = sum(p.get('tx_mbps', 0.0) for p in state_manager.port_rates.values())
                total_pps = sum(p.get('tx_pps', 0.0) for p in state_manager.port_rates.values())

                flows = []
                for (dpid, src_ip, dst_ip), stat in list(state_manager.flow_stats.items())[:15]:
                    flows.append({
                        'switch': dpid,
                        'src': src_ip,
                        'dst': dst_ip,
                        'packets': stat['packets'],
                        'bytes': stat['bytes'],
                        'mbps': round(stat.get('mbps', 0.0), 2),
                        'pps': round(stat.get('pps', 0.0), 1)
                    })

                return {
                    'total_throughput_mbps': round(total_mbps, 2),
                    'total_packet_rate_pps': round(total_pps, 1),
                    'active_flows_count': len(state_manager.flow_stats),
                    'flows': flows
                }

            def _get_security_data(self):
                sec_state = state_manager.get_security_state()
                return {
                    'packet_rate_pps': round(float(sec_state[0]) * 5000.0, 1),
                    'shannon_entropy': round(float(sec_state[1]), 3),
                    'bytes_per_packet': round(float(sec_state[2]) * 1500.0, 1),
                    'active_flows': int(float(sec_state[3]) * 100.0),
                    'bandwidth_intensity': round(float(sec_state[4]), 3),
                    'blocked_ips': [str(ip) for ip in controller.security_module.blocked_ips],
                    'ddos_alert': bool(len(controller.security_module.blocked_ips) > 0 or float(sec_state[1]) < 0.4)
                }

            def _get_rl_data(self):
                return {
                    'router_epsilon': round(float(controller.routing_module.agent.epsilon), 3),
                    'multicast_epsilon': round(float(controller.multicast_module.agent.epsilon), 3),
                    'security_noise': round(float(controller.security_module.agent.noise_std), 3),
                    'router_device': str(controller.routing_module.agent.device),
                    'multicast_device': str(controller.multicast_module.agent.device),
                    'security_device': str(controller.security_module.agent.device)
                }

            def log_message(self, format, *args):
                pass # Suppress HTTP access logging in controller console

        try:
            server = HTTPServer(('0.0.0.0', self.port), RequestHandler)
            self.logger.info("=" * 65)
            self.logger.info(f" Web Dashboard & REST API active at http://localhost:{self.port}")
            self.logger.info("=" * 65)
            server.serve_forever()
        except Exception as e:
            self.logger.error(f"[DashboardServer] Failed to start on port {self.port}: {e}")
