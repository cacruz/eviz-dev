from eviz.lib.data.url_validator import is_url, is_opendap_url


class TestUrlValidator:
    def test_is_url_with_valid_urls(self):
        """Test is_url with valid URLs."""
        valid_urls = [
            "http://example.com",
            "https://example.com",
            "https://example.com/path/to/file.nc",
            "https://example.com/path/to/file.nc?query=param",
            "https://psl.noaa.gov/thredds/dodsC/Datasets/ncep.reanalysis.derived/surface/air.mon.mean.nc"
        ]
        
        for url in valid_urls:
            assert is_url(url) is True, f"Expected {url} to be recognized as a URL"
    
    def test_is_url_with_invalid_urls(self):
        """Test is_url with invalid URLs."""
        invalid_urls = [
            "",
            "not a url",
            "/path/to/file.nc",
            "./relative/path.txt",
            "C:\\Windows\\path.txt",
            "file.txt",
            "http:",
            "http://"
        ]
        
        for url in invalid_urls:
            assert is_url(url) is False, f"Expected {url} to be recognized as not a URL"
    
    def test_is_opendap_url_with_valid_opendap_urls(self):
        """Test is_opendap_url with valid OpenDAP URLs."""
        valid_opendap_urls = [
            "https://psl.noaa.gov/thredds/dodsC/Datasets/ncep.reanalysis.derived/surface/air.mon.mean.nc",
            "http://example.com/thredds/dodsC/path/to/file.nc",
            "https://example.com/opendap/path/to/file.nc",
            "https://example.com/dods/path/to/file.nc",
            "https://example.com/dap/path/to/file.nc",
            "https://example.com/path/to/file.nc?dataset=value"
        ]
        
        for url in valid_opendap_urls:
            assert is_opendap_url(url) is True, f"Expected {url} to be recognized as an OpenDAP URL"
    
    def test_is_opendap_url_with_invalid_opendap_urls(self):
        """Test is_opendap_url with invalid OpenDAP URLs."""
        invalid_opendap_urls = [
            "",
            "not a url",
            "/path/to/file.nc",
            "http://example.com/path/to/file.txt",
            "https://example.com/path/to/file.pdf",
        ]
        
        for url in invalid_opendap_urls:
            assert is_opendap_url(url) is False, f"Expected {url} to be recognized as not an OpenDAP URL"
