import unittest
from unittest.mock import patch
from urllib.error import HTTPError, URLError

from sts_lab.runtime import probe_model_access


class AccessTests(unittest.TestCase):
    def test_cli_saved_token_is_used_without_being_reported(self):
        from pathlib import Path
        import tempfile
        from sts_lab.runtime import saved_hf_token
        with tempfile.TemporaryDirectory() as directory:
            Path(directory,'token').write_text('local-test-credential')
            with patch.dict('os.environ',{'HF_HOME':directory,'HF_TOKEN':''},clear=False):
                self.assertEqual(saved_hf_token(),'local-test-credential')
    def test_access_failure_is_explicit_and_secret_free(self):
        error=HTTPError('https://example.invalid',401,'secret response',{},None)
        with patch('urllib.request.urlopen',side_effect=error),patch.dict('os.environ',{'HF_TOKEN':'sensitive-test-value'}):
            result=probe_model_access()
        self.assertEqual(result['reason'],'model_access_required')
        self.assertNotIn('sensitive',str(result))
        self.assertNotIn('secret',str(result))

    def test_missing_asset_and_network_failure_distinguished(self):
        for error,reason in ((HTTPError('https://example.invalid',404,'',{},None),'asset_unavailable'),
                             (URLError('offline'),'network_unavailable')):
            with patch('urllib.request.urlopen',side_effect=error):
                self.assertEqual(probe_model_access()['reason'],reason)
