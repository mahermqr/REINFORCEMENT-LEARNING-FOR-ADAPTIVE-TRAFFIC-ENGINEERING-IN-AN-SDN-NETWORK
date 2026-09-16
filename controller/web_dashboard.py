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
    Lightweight Embedded HTTP REST Server & Real-Time Web Dashboard for Ryu (EC499).
    Serves live network topology, port/flow metrics, RL agent telemetry,
    latency, jitter, packet loss, and control overhead at http://localhost:8080.
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
                elif p == '/api/control_overhead':
                    self._send_json(state_manager.get_control_overhead_summary())
                elif p == '/api/rl_metrics':
                    self._send_json(self._get_rl_data())
                elif p == '/api/history':
                    self._send_json(list(state_manager.telemetry_history))
                elif p == '/api/benchmarks':
                    self._send_json(self._get_benchmark_data())
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
                elif p == '/api/simulate/switch_topology':
                    topo_id = query.get('topo', ['tree'])[0]
                    msg = self._handle_switch_topology(topo_id)
                    self._send_json({"status": "success", "message": msg, "current_topo": topo_id})
                elif p == '/api/simulate/reset':
                    state_manager.reset_simulation()
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
                    super().do_GET()

            def do_POST(self):
                parsed = urlparse(self.path)
                p = parsed.path
                if p == '/api/simulate/traffic_burst':
                    self._handle_simulate_traffic_burst()
                    self._send_json({"status": "success", "message": "⚡ Traffic burst injected."})
                elif p == '/api/simulate/core_jamming':
                    state_manager.inject_core_congestion(utilization=0.96)
                    self._send_json({"status": "success", "message": "🔥 Core links saturated. DQN offload active."})
                elif p == '/api/simulate/reset':
                    state_manager.reset_simulation()
                    self._send_json({"status": "success", "message": "🔄 Nominal state restored."})
                else:
                    self.send_response(404)
                    self.end_headers()

            def _send_json(self, data):
                self.send_response(200)
                self.send_header('Content-Type', 'application/json')
                self.send_header('Access-Control-Allow-Origin', '*')
                self.end_headers()
                self.wfile.write(json.dumps(data, default=numpy_json_serializer).encode('utf-8'))

            def _handle_simulate_traffic_burst(self):
                paths_to_test = [
                    ("10.0.0.2", "10.0.0.5", 4, 6, [4, 6]),
                    ("10.0.0.3", "10.0.0.7", 5, 7, [5, 7]),
                    ("10.0.0.1", "10.0.0.6", 4, 6, [4, 2, 1, 3, 6])
                ]
                for src_ip, dst_ip, src_d, dst_d, p in paths_to_test:
                    state_manager.inject_traffic_flow(src_ip, dst_ip, src_d, dst_d, mbps=22.5, pps=1600.0, path=p)

            def _handle_switch_topology(self, topo_id):
                builder = ALL_TOPOLOGY_BUILDERS.get(topo_id)
                if not builder:
                    return f"Unknown topology: {topo_id}"
                g_new, meta = builder()
                state_manager.graph = g_new.copy()
                state_manager.link_bandwidths.clear()
                state_manager.link_delays.clear()
                state_manager.link_jitter.clear()
                state_manager.link_loss.clear()
                state_manager.link_utilization.clear()
                state_manager._routing_path_cache.clear()

                for u, v, d in g_new.edges(data=True):
                    cap = d.get('capacity', 100.0)
                    lat = d.get('delay', 2.0)
                    state_manager.update_link(u, v, src_port=1, dst_port=1, capacity_mbps=cap, delay_ms=lat)
                return f"Switched to {meta.get('name', topo_id)}"

            def _get_topology_data(self, topo_id=None):
                if topo_id and topo_id in ALL_TOPOLOGY_BUILDERS:
                    g_req, meta_req = ALL_TOPOLOGY_BUILDERS[topo_id]()
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
                            'cpu': 0.1,
                            'ram': 0.2
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
                        'cpu': 0.1,
                        'ram': 0.2
                    })

                links = []
                for u, v in state_manager.graph.edges():
                    if u < v or not state_manager.graph.has_edge(v, u):
                        links.append({
                            'source': u,
                            'target': v,
                            'utilization': round(state_manager.link_utilization.get((u, v), 0.0) * 100, 1),
                            'delay': round(state_manager.link_delays.get((u, v), 2.0), 2),
                            'jitter': round(state_manager.link_jitter.get((u, v), 0.25), 2),
                            'loss_pct': round(state_manager.link_loss.get((u, v), 0.0), 2),
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
                return state_manager.get_network_te_summary()

            def _get_rl_data(self):
                return {
                    'router_epsilon': round(float(controller.routing_module.agent.epsilon), 3),
                    'router_device': str(controller.routing_module.agent.device),
                    'use_per': bool(controller.routing_module.agent.use_per),
                    'memory_size': len(controller.routing_module.agent.memory)
                }

            def _get_benchmark_data(self):
                json_path = os.path.join(BASE_DIR, 'logs', 'routing_tournament_results.json')
                if os.path.exists(json_path):
                    try:
                        with open(json_path, 'r') as f:
                            return json.load(f)
                    except Exception:
                        pass
                return {"message": "Tournament benchmark pending execution"}

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
