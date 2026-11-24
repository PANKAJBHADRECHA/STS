from http.client import HTTPConnection
import json
import threading
import unittest

from sts_lab.server import create_server


class ServerTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.server=create_server(0)
        cls.thread=threading.Thread(target=cls.server.serve_forever,daemon=True)
        cls.thread.start()
        cls.port=cls.server.server_port

    @classmethod
    def tearDownClass(cls):
        cls.server.shutdown();cls.server.server_close();cls.thread.join()

    def request(self,path="/",method="GET",body=None,headers=None):
        connection=HTTPConnection("127.0.0.1",self.port,timeout=10)
        connection.request(method,path,body=body,headers=headers or {})
        response=connection.getresponse();status=response.status;result=response.read();meta=dict(response.getheaders());connection.close()
        return status,result,meta

    def local_headers(self):
        return {"Origin":f"http://127.0.0.1:{self.port}","Content-Type":"application/json"}

    def test_ui_and_assets(self):
        for path in ("/","/app.js","/style.css","/api/status","/api/scenarios"):
            with self.subTest(path=path):
                status,body,headers=self.request(path)
                self.assertEqual(status,200)
                self.assertTrue(body)
                self.assertIn("Content-Security-Policy",headers)

    def test_dns_rebinding_is_rejected(self):
        self.assertEqual(self.request(headers={"Host":"attacker.invalid"})[0],403)

    def test_cross_origin_and_missing_origin_cannot_mutate(self):
        for headers in ({}, {"Origin":"https://attacker.invalid"}):
            self.assertEqual(self.request("/api/evaluate","POST","{}",headers)[0],403)

    def test_bad_json_and_oversize_requests(self):
        for body in ("not-json","[]",'"text"'):
            self.assertEqual(self.request("/api/replay","POST",body,self.local_headers())[0],400)
        self.assertEqual(self.request("/api/replay","POST","x"*4097,self.local_headers())[0],413)

    def test_replay_runs_each_family(self):
        from sts_lab.scenarios import FAMILIES
        for family in FAMILIES:
            status,body,_=self.request("/api/replay","POST",json.dumps({"family":family,"variant":"combined","index":0}),self.local_headers())
            self.assertEqual(status,200)
            self.assertTrue(json.loads(body)["trace"])

    def test_replay_rejects_unknown_or_wrong_typed_inputs(self):
        for index in (-1,20,1.5,True,"0"):
            payload=json.dumps({"family":"interruptions","variant":"combined","index":index})
            self.assertEqual(self.request("/api/replay","POST",payload,self.local_headers())[0],400)

    def test_path_traversal_and_unknown_routes(self):
        for path in ("/../.env","/api/document?name=../../.env","/nope"):
            self.assertEqual(self.request(path)[0],404)
