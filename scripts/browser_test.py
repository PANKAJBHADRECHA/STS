"""Real Chromium QA: all 25 policy/family combinations, four cases each."""
import json
from pathlib import Path
import sys
import threading

ROOT=Path(__file__).resolve().parent.parent
sys.path.insert(0,str(ROOT))
from sts_lab.server import create_server
from sts_lab.core import POLICIES
from sts_lab.scenarios import FAMILIES


def main():
    from playwright.sync_api import sync_playwright, expect
    server=create_server(0)
    thread=threading.Thread(target=server.serve_forever,daemon=True);thread.start()
    url=f"http://127.0.0.1:{server.server_port}"
    errors=[];runs=[]
    try:
        with sync_playwright() as p:
            browser=p.chromium.launch()
            page=browser.new_page(viewport={"width":1440,"height":1050})
            page.on("pageerror",lambda e:errors.append(str(e)))
            page.on("console",lambda m:errors.append(m.text) if m.type=="error" else None)
            page.goto(url);expect(page.locator("tbody tr")).to_have_count(5)
            with page.expect_response(lambda r:r.url.endswith('/api/evaluate') and r.request.method=='POST') as response:
                page.locator('#run').click()
            assert response.value.status==200
            expect(page.locator('#status')).to_contain_text('Evaluation complete')
            chosen=response.value.json()['selection']['winner']
            assert chosen is not None
            assert page.locator('tr.winner .variant-name').inner_text().startswith(chosen.replace('_',' ').capitalize())
            assert page.locator('#variant').input_value()==chosen
            page.screenshot(path=str(ROOT/'reports/desktop.png'),full_page=True)
            page.get_by_role('button',name='Scenario replay',exact=True).click()
            for policy in POLICIES:
                page.locator('#variant').select_option(policy)
                for family in FAMILIES:
                    page.locator('#family').select_option(family)
                    for index in range(4):
                        page.locator('#case-index').fill(str(index))
                        with page.expect_response(lambda r:r.url.endswith('/api/replay')) as response:
                            page.locator('#replay-run').click()
                        data=response.value.json()
                        expect(page.locator('#replay-run')).to_be_enabled()
                        expect(page.locator('#replay-result .trace-item')).to_have_count(len(data['trace']))
                        assert data['variant']==policy and data['family']==family
                        runs.append({"variant":policy,"family":family,"case":index,"passed":data['passed']})
            for label in ('Research dossier','Build memo','Evaluation notes'):
                page.get_by_role('button',name=label,exact=True).click()
                expect(page.locator('#document-body')).not_to_have_text('Loading document…')
                assert len(page.locator('#document-body').inner_text())>500
            page.get_by_role('button',name='Experiment',exact=True).click()
            with page.expect_download() as download:
                page.locator('#export').click()
            assert download.value.suggested_filename.endswith('.json')
            page.set_viewport_size({"width":390,"height":844})
            page.screenshot(path=str(ROOT/'reports/mobile.png'),full_page=True)
            assert page.evaluate('document.documentElement.scrollWidth <= window.innerWidth')
            assert not errors, errors
            browser.close()
        report={"browser":"Chromium 141 / Playwright 1.56.0","replays":len(runs),"combinations":25,
                "desktop":"1440x1050","mobile":"390x844","console_errors":errors,
                "documents_checked":3,"download_checked":True,"runs":runs,"status":"passed"}
        (ROOT/'reports/browser.json').write_text(json.dumps(report,indent=2)+'\n')
        print(json.dumps({k:v for k,v in report.items() if k!='runs'},indent=2))
    finally:
        server.shutdown();server.server_close();thread.join()


if __name__=='__main__':main()
