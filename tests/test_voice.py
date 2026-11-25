import array
import io
import json
import threading
import unittest
import wave
from http.client import HTTPConnection
from unittest.mock import patch

from sts_lab.server import create_server


def pcm_wav(amplitude=1000,seconds=.25):
    output=io.BytesIO()
    with wave.open(output,'wb') as stream:
        stream.setnchannels(1);stream.setsampwidth(2);stream.setframerate(24000)
        stream.writeframes(array.array('h',[amplitude]*int(24000*seconds)).tobytes())
    return output.getvalue()


class VoiceAPITests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.server=create_server(0);cls.port=cls.server.server_port
        cls.thread=threading.Thread(target=cls.server.serve_forever,daemon=True);cls.thread.start()

    @classmethod
    def tearDownClass(cls):
        cls.server.shutdown();cls.server.server_close();cls.thread.join()

    def post(self,path,body=b'{}',session='12345678-1234-1234-1234-123456789abc',mime='application/json',origin=True,length=None):
        connection=HTTPConnection('127.0.0.1',self.port,timeout=5)
        headers={'Content-Type':mime,'X-STS-Session':session}
        if length is not None:headers['Content-Length']=str(length)
        if origin:headers['Origin']=f'http://127.0.0.1:{self.port}'
        connection.request('POST',path,body,headers);response=connection.getresponse();status=response.status;data=json.loads(response.read());connection.close();return status,data

    def test_microphone_permission_and_audio_csp(self):
        connection=HTTPConnection('127.0.0.1',self.port)
        connection.request('GET','/');response=connection.getresponse()
        self.assertEqual(response.getheader('Permissions-Policy'),'microphone=(self), camera=()')
        self.assertIn("media-src 'self' blob:",response.getheader('Content-Security-Policy'))
        response.read();connection.close()

    def test_no_cross_origin_capture_requests(self):
        self.assertEqual(self.post('/api/voice/load',origin=False)[0],403)

    def test_session_id_required(self):
        self.assertEqual(self.post('/api/voice/load',session='bad')[0],400)

    def test_model_load_is_explicit_and_backend_error_visible(self):
        with patch('sts_lab.server.VOICE.load',return_value={'ready':True}):
            self.assertEqual(self.post('/api/voice/load'),(200,{'ready':True}))
        with patch('sts_lab.server.VOICE.load',side_effect=RuntimeError('assets missing')):
            status,data=self.post('/api/voice/load');self.assertEqual(status,503);self.assertEqual(data['error'],'assets missing')

    def test_turn_payload_and_reset_forwarded(self):
        blob=pcm_wav()
        with patch('sts_lab.server.VOICE.turn',return_value={'text':'hello','audio_base64':'example'}) as call:
            status,data=self.post('/api/voice/turn?reset=1',blob,mime='audio/wav')
            self.assertEqual(status,200);self.assertEqual(data['text'],'hello')
            self.assertEqual(call.call_args.args[0],blob);self.assertTrue(call.call_args.kwargs['reset'])

    def test_bad_media_size_and_format(self):
        self.assertEqual(self.post('/api/voice/turn',b'bad',mime='audio/wav')[0],413)
        self.assertEqual(self.post('/api/voice/turn',pcm_wav(),mime='application/json')[0],415)
        self.assertEqual(self.post('/api/voice/turn',b'x'*44,mime='audio/wav',length=2_000_001)[0],413)

    def test_busy_and_timeout_are_distinct(self):
        for exception,status in ((BlockingIOError('busy'),409),(TimeoutError('timeout'),504)):
            with patch('sts_lab.server.VOICE.turn',side_effect=exception):
                self.assertEqual(self.post('/api/voice/turn',pcm_wav(),mime='audio/wav')[0],status)

    def test_silence_rejected_before_model_load_and_temp_files_removed(self):
        from pathlib import Path
        from sts_lab.voice_service import VoiceService,ROOT
        service=VoiceService()
        before=set((ROOT/'.runtime').glob('sts-voice-*'))
        try:
            with patch.object(service,'_load') as load:
                with self.assertRaisesRegex(ValueError,'No audible speech'):
                    service.turn(pcm_wav(amplitude=0),'session')
                load.assert_not_called()
            self.assertEqual(set((ROOT/'.runtime').glob('sts-voice-*')),before)
        finally:service.close()
