"""Test actual native inference through upload and a browser microphone stream.

The microphone is Chromium's fixture-backed capture device, not the user's
physical microphone. No inference responses are mocked.
"""
import array
import base64
import json
from pathlib import Path
import subprocess
import sys
import threading
import time
import wave

ROOT=Path(__file__).resolve().parent.parent
sys.path.insert(0,str(ROOT))
from sts_lab.server import create_server
from sts_lab.voice_service import VOICE
from sts_lab.audio import inspect_wav


def main():
    from playwright.sync_api import sync_playwright,expect
    fixtures=ROOT/'.runtime/conversation-fixtures'
    fake=fixtures/'browser-microphone.wav'
    subprocess.run(['afconvert','-f','WAVE','-d','LEI16@48000',str(fixtures/'greeting.wav'),str(fake)],check=True)
    with wave.open(str(fake),'rb') as stream:
        raw=stream.readframes(stream.getnframes())
    with wave.open(str(fake),'wb') as stream:
        stream.setnchannels(1);stream.setsampwidth(2);stream.setframerate(48000);stream.writeframes(raw+bytes(48000*2*3))
    server=create_server(0);thread=threading.Thread(target=server.serve_forever,daemon=True);thread.start()
    url=f'http://127.0.0.1:{server.server_port}';errors=[];cases=[]
    try:
        with sync_playwright() as p:
            browser=p.chromium.launch(args=['--use-fake-device-for-media-stream','--use-fake-ui-for-media-stream',f'--use-file-for-fake-audio-capture={fake}','--autoplay-policy=no-user-gesture-required'])
            context=browser.new_context(permissions=['microphone'],viewport={'width':1440,'height':1000})
            page=context.new_page();page.on('pageerror',lambda e:errors.append(str(e)))
            page.add_init_script("window.__voiceReplies=[];window.addEventListener('sts-voice-reply',e=>window.__voiceReplies.push(e.detail));")
            page.on('console',lambda m:errors.append(m.text) if m.type=='error' else None)
            page.set_default_timeout(120000);page.goto(url)
            expect(page.locator('#voice')).to_be_visible()
            with page.expect_response(lambda r:r.url.endswith('/api/voice/load'),timeout=120000) as response:
                page.locator('#voice-load').click()
            assert response.value.status==200,response.value.text()
            expect(page.locator('#record')).to_be_enabled()
            print('Browser loaded real native model',flush=True)
            with page.expect_response(lambda r:'/api/voice/turn' in r.url,timeout=120000) as response:
                page.locator('#voice-file').set_input_files(str(fixtures/'arithmetic.wav'))
            assert response.value.status==200
            expect(page.locator('#record')).to_be_enabled()
            page.wait_for_function('() => window.__voiceReplies.length === 1')
            data=page.evaluate('window.__voiceReplies[0]')
            assert data['text'] and data['output']['rms']>.001
            cases.append({'path':'WAV upload','text':data['text'],'wall_seconds':data['wall_seconds'],'turn':data['turn']})
            (fixtures/'browser-upload-reply.wav').write_bytes(base64.b64decode(data['audio_base64']))
            print('Real upload response:',data['text'],flush=True)
            with page.expect_response(lambda r:r.url.endswith('/api/voice/reset')):
                page.locator('#voice-new').click()
            page.locator('#record').click()
            try:
                expect(page.locator('#record')).to_have_text('Stop and send',timeout=15000)
            except AssertionError:
                raise RuntimeError('Microphone start failed: '+page.locator('#voice-state').inner_text()) from None
            page.wait_for_function("() => document.getElementById('mic-level').value > 0.01",timeout=10000)
            page.wait_for_timeout(2500)
            with page.expect_response(lambda r:'/api/voice/turn' in r.url,timeout=120000) as response:
                page.locator('#record').click()
            assert response.value.status==200
            expect(page.locator('#record')).to_be_enabled()
            page.wait_for_function('() => window.__voiceReplies.length === 2')
            data=page.evaluate('window.__voiceReplies[1]')
            assert data['text'] and data['input_seconds']>.5 and data['output']['rms']>.001
            assert data['turn']==1
            cases.append({'path':'Browser microphone capture (fixture device)','text':data['text'],'input_seconds':data['input_seconds'],'wall_seconds':data['wall_seconds'],'turn':data['turn']})
            (fixtures/'browser-mic-reply.wav').write_bytes(base64.b64decode(data['audio_base64']))
            expect(page.locator('#voice-messages audio')).to_have_count(1)
            page.wait_for_function("() => document.querySelector('#voice-messages audio').readyState >= 2")
            page.screenshot(path=str(ROOT/'reports/voice-desktop.png'),full_page=True)
            page.set_viewport_size({'width':390,'height':844});page.screenshot(path=str(ROOT/'reports/voice-mobile.png'),full_page=True)
            assert page.evaluate('document.documentElement.scrollWidth <= innerWidth')
            page.locator('#voice-stop').click()
            assert not errors,errors
            browser.close()
        report={'status':'passed','inference_mocked':False,'physical_microphone_tested':False,'capture_device':'Chromium WAV-backed microphone',
                'cases':cases,'audio_playback_decoded':True,'console_errors':errors,'desktop':'1440x1000','mobile':'390x844'}
        (ROOT/'reports/voice-browser.json').write_text(json.dumps(report,indent=2)+'\n')
        print(json.dumps(report,indent=2))
    finally:
        VOICE.close();server.shutdown();server.server_close();thread.join()


if __name__=='__main__':main()
