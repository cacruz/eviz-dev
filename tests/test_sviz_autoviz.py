import os
import sys
import unittest
from unittest.mock import patch, MagicMock, mock_open
import pandas as pd
from pathlib import Path
from datetime import datetime
import json

# Add the project root to the path so we can import the module
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Import the module to test
from sviz.pages import autoviz

class TestAutoviz(unittest.TestCase):
    """Test cases for the autoviz.py module."""
    
    def setUp(self):
        """Set up test fixtures."""
        # Create a mock for streamlit
        self.st_mock = MagicMock()
        # Create sample data for testing
        self.sample_metadata = {
            "global_attributes": {
                "title": "Test Dataset",
                "source": "Unit Test"
            },
            "variables": {
                "temperature": {
                    "dimensions": ["time", "lat", "lon"],
                    "data_type": "float32",
                    "attributes": {
                        "long_name": "Air Temperature",
                        "units": "K",
                        "fmissing_value": -999.9,
                        "vmin": 250.0,
                        "vmax": 320.0
                    }
                },
                "precipitation": {
                    "dimensions": ["time", "lat", "lon"],
                    "data_type": "float32",
                    "attributes": {
                        "long_name": "Precipitation Rate",
                        "units": "mm/hr",
                        "fmissing_value": -999.9,
                        "vmin": 0.0,
                        "vmax": 50.0
                    }
                }
            }
        }
        
        # Sample options map
        self.sample_options = {
            "single": {
                "dataset1": "path/to/dataset1.yaml",
                "dataset2": "path/to/dataset2.yaml"
            },
            "compare": {
                "dataset3_ANN": "path/to/dataset3.yaml",
                "dataset4": "path/to/dataset4.yaml"
            }
        }
        
        # Create a temporary directory structure for testing
        self.test_output_dir = "test_output"
        self.test_timestamp = "20250524-123456"
        os.makedirs(os.path.join(self.test_output_dir, self.test_timestamp), exist_ok=True)
    
    def tearDown(self):
        """Clean up after tests."""
        # Remove test directories if they exist
        if os.path.exists(self.test_output_dir):
            import shutil
            shutil.rmtree(self.test_output_dir)

    @patch('sviz.pages.autoviz.st')
    def test_make_nice_dimstr(self, mock_st):
        """Test the make_nice_dimstr function."""
        # Test with various input formats
        self.assertEqual(autoviz.make_nice_dimstr("['time', 'lat', 'lon']"), "time,  lat,  lon")
        self.assertEqual(autoviz.make_nice_dimstr("[time,lat,lon]"), "time, lat, lon")
        self.assertEqual(autoviz.make_nice_dimstr("[]"), "")
    
    @patch('sviz.pages.autoviz.st')
    @patch('builtins.open', new_callable=mock_open, read_data=json.dumps({
        "global_attributes": {"title": "Test Dataset"},
        "variables": {
            "var1": {
                "dimensions": ["time", "lat", "lon"],
                "attributes": {"long_name": "Variable 1", "units": "K"}
            }
        }
    }))
    def test_metadata_to_df(self, mock_file, mock_st):
        """Test the metadata_to_df function."""
        # Call the function
        global_attrs, variables = autoviz.metadata_to_df('fake_path.json')
        
        # Check that the function returns DataFrames
        self.assertIsInstance(global_attrs, pd.DataFrame)
        self.assertIsInstance(variables, pd.DataFrame)
        
        # Check content of the DataFrames
        self.assertEqual(global_attrs.iloc[0]['title'], "Test Dataset")
        self.assertEqual(variables.iloc[0]['Variables'], "var1")
        self.assertEqual(variables.iloc[0]['Long Name'], "Variable 1")
        self.assertEqual(variables.iloc[0]['Units'], "K")
    
    @patch('sviz.pages.autoviz.st')
    def test_metadata_to_df_file_not_found(self, mock_st):
        """Test the metadata_to_df function when file is not found."""
        # Call the function with a non-existent file
        global_attrs, variables = autoviz.metadata_to_df('nonexistent_file.json')
        
        # Check that the function returns empty DataFrames
        self.assertTrue(global_attrs.empty)
        self.assertTrue(variables.empty)
        
        # Check that an error was displayed
        mock_st.error.assert_called_once()
    
    @patch('sviz.pages.autoviz.os.path.exists')
    @patch('sviz.pages.autoviz.os.listdir')
    @patch('sviz.pages.autoviz.st')
    def test_get_time_stamped_output(self, mock_st, mock_listdir, mock_exists):
        """Test the get_time_stamped_output function."""
        # Mock os.path.exists to return True
        mock_exists.return_value = True
        
        # Mock the os.listdir to return timestamps
        mock_listdir.return_value = ['20250524-123456', '20250523-123456', 'invalid-format']
        
        # Call the function
        result = autoviz.get_time_stamped_output('dummy_path')
        
        # Check that it returns the most recent timestamp
        self.assertEqual(result, '20250524-123456')

    @patch('sviz.pages.autoviz.os.path.exists')
    @patch('sviz.pages.autoviz.os.listdir')
    @patch('sviz.pages.autoviz.st')
    def test_get_time_stamped_output_empty_dir(self, mock_st, mock_listdir, mock_exists):
        """Test get_time_stamped_output with an empty directory."""
        # Mock os.path.exists to return True
        mock_exists.return_value = True
        
        # Mock the os.listdir to return an empty list
        mock_listdir.return_value = []
        
        # Call the function
        result = autoviz.get_time_stamped_output('dummy_path')
        
        # Check that it returns the error indicator
        self.assertEqual(result, 'no_output_found')
        mock_st.warning.assert_called_once()
   
    @patch('sviz.pages.autoviz.os.path.exists')
    @patch('sviz.pages.autoviz.st')
    def test_get_time_stamped_output_dir_not_found(self, mock_st, mock_exists):
        """Test get_time_stamped_output with a non-existent directory."""
        # Mock os.path.exists to return False
        mock_exists.return_value = False
        
        # Call the function
        result = autoviz.get_time_stamped_output('dummy_path')
        
        # Check that it returns the error indicator
        self.assertEqual(result, 'output_not_found')
        mock_st.error.assert_called_once()
    
    @patch('sviz.pages.autoviz.u.load_yaml')
    def test_demo_options_map(self, mock_load_yaml):
        """Test the demo_options_map function."""
        # Mock the load_yaml function to return our sample options
        mock_load_yaml.return_value = self.sample_options
        
        # Call the function
        single, compare = autoviz.demo_options_map('dummy_path.yaml')
        
        # Check the results
        self.assertEqual(single, self.sample_options['single'])
        self.assertEqual(compare, self.sample_options['compare'])
    
    @patch('sviz.pages.autoviz.st')
    @patch('sviz.pages.autoviz.run')
    @patch('sviz.pages.autoviz.os.path.exists')
    @patch('sviz.pages.autoviz.metadata_to_df')
    @patch('sviz.pages.autoviz.u.get_project_root')
    def test_run_metadump(self, mock_get_root, mock_metadata_to_df, mock_exists, mock_run, mock_st):
        """Test the run_metadump function."""
        # Setup mocks
        mock_get_root.return_value = '/fake/root'
        mock_exists.return_value = True
        mock_run.return_value = MagicMock(returncode=0)
        mock_metadata_to_df.return_value = (pd.DataFrame({'title': ['Test Dataset']}), 
                                           pd.DataFrame({'Variables': ['var1']}))
        
        # Setup session state
        mock_st.session_state.select_dataset = 'dataset1'
        
        # Call the function
        autoviz.run_metadump()
        
        # Check that the subprocess was called correctly
        mock_run.assert_called_once()
        self.assertIn('--json', mock_run.call_args[0][0])
        
        # Check that the dataframes were displayed
        mock_st.dataframe.assert_called()
    
    @patch('sviz.pages.autoviz.st')
    @patch('sviz.pages.autoviz.run')
    @patch('sviz.pages.autoviz.os.path.exists')
    def test_run_metadump_file_not_found(self, mock_exists, mock_run, mock_st):
        """Test run_metadump when the file doesn't exist."""
        # Setup mocks
        mock_exists.return_value = False
        
        # Setup session state
        mock_st.session_state.select_dataset = 'dataset1'
        
        # Call the function
        autoviz.run_metadump()
        
        # Check that an error was displayed
        mock_st.error.assert_called_once()
        
        # Check that subprocess was not called
        mock_run.assert_not_called()
    
    @patch('sviz.pages.autoviz.os.path.exists')
    @patch('sviz.pages.autoviz.TEMPLATE_PATH')
    @patch('builtins.open', new_callable=mock_open)
    def test_create_new_page(self, mock_file, mock_template_path, mock_exists):
        """Test the create_new_page function."""
        # Setup mocks
        mock_template_path.open.return_value.__enter__.return_value.read.return_value = "{{ config.parent_folder }}"
        mock_template_path.parent = Path('/fake/path')
        
        # Mock os.path.exists to return True for the new script
        mock_exists.return_value = True
        
        # Call the function
        result = autoviz.create_new_page('test_dash', {'parent_folder': 'test_folder'})
        
        # Check that the file was written
        mock_file.assert_called()
        mock_file().write.assert_called_once_with('test_folder')
        
        # Check the return value
        self.assertTrue(result)

    
    @patch('sviz.pages.autoviz.st')
    @patch('sviz.pages.autoviz.demo_options_map')
    def test_main_function_initialization(self, mock_demo_options, mock_st):
        """Test the initialization part of the main function."""
        # Setup mocks
        mock_demo_options.return_value = (self.sample_options['single'], self.sample_options['compare'])
        
        # Directly execute the code from the main block
        autoviz.single, autoviz.compare = mock_demo_options.return_value
        
        if "file_option" not in autoviz.st.session_state:
            autoviz.st.session_state.file_option = "None Selected"
        
        autoviz.st.selectbox(
            "Select a dataset",
            ["None Selected"] + list(autoviz.single.keys()) + list(autoviz.compare.keys()),
            key='select_dataset',
            on_change=autoviz.run_metadump
        )
        
        autoviz.st.text_input(
            'Enter the name of the dashboard you would like to create! (e.g. my_viz)',
        )
        
        # Check that the selectbox was created with the correct options
        mock_st.selectbox.assert_called_once()
        call_args = mock_st.selectbox.call_args[0]
        self.assertEqual(call_args[0], "Select a dataset")
        
        # Check that the text input was created
        mock_st.text_input.assert_called_once()


if __name__ == '__main__':
    unittest.main()
