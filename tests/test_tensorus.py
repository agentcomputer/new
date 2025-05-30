import unittest
import os
import shutil
import uuid
import numpy as np
import json
from tensorus import client # Assuming tensorus/client.py is accessible

# Configuration constants (can be imported or redefined for tests)
HDF5_DIR_TEST = "./tensorus_tensors_test"
ANN_INDEX_DIR_TEST = "./tensorus_indices_test"

# Original DB config (tests would ideally use a separate test DB)
DB_HOST_TEST = "localhost"
DB_NAME_TEST = "tensorus_db_test" # Ideally a separate test database
DB_USER_TEST = "tensorus_user"
DB_PASSWORD_TEST = "tensorus_password"


class TestTensorusAPI(unittest.TestCase):
    """
    Test suite for the Tensorus client API.

    This suite covers basic tensor operations (create, get, update, delete)
    and index operations. It is designed to run against a test database
    and uses separate directories for HDF5 files and ANN indices created
    during tests.

    Note: Many tests require a running PostgreSQL database configured as per
    DB_HOST_TEST, DB_NAME_TEST, etc. If the database is not available,
    tests relying on it are expected to fail.
    """

    @classmethod
    def setUpClass(cls):
        """
        Sets up the test environment before any tests are run.

        This method performs the following setup actions:
        - Overrides the default HDF5 and ANN index directory paths in the
          `tensorus.client` module to use test-specific directories.
        - Overrides the default database connection parameters in the
          `tensorus.client` module to use a test database configuration.
        - Creates the test-specific HDF5 and ANN index directories.
        - Attempts to initialize the test database schema by calling
          `client.init_db()`. A warning is printed if this fails.
        """
        # Override paths for testing
        cls.original_hdf5_dir = client.HDF5_DIR
        cls.original_ann_index_dir = client.ANN_INDEX_DIR
        client.HDF5_DIR = HDF5_DIR_TEST
        client.ANN_INDEX_DIR = ANN_INDEX_DIR_TEST

        # Override DB config for testing (IMPORTANT: This assumes client.py uses these global vars directly)
        # A better way would be to pass config to functions or use a config object.
        cls.original_db_host = client.DB_HOST
        cls.original_db_name = client.DB_NAME
        cls.original_db_user = client.DB_USER
        cls.original_db_password = client.DB_PASSWORD
        client.DB_HOST = DB_HOST_TEST
        client.DB_NAME = DB_NAME_TEST
        client.DB_USER = DB_USER_TEST
        client.DB_PASSWORD = DB_PASSWORD_TEST

        # Create test directories
        os.makedirs(HDF5_DIR_TEST, exist_ok=True)
        os.makedirs(ANN_INDEX_DIR_TEST, exist_ok=True)
        
        # NOTE: Database initialization (client.init_db()) would typically be called here.
        # However, it will fail if DB is not available.
        # For now, we'll proceed assuming it might be run in an env with a DB.
        try:
            client.init_db()
            print(f"Test database {DB_NAME_TEST} initialized (or tables checked).")
        except Exception as e:
            print(f"WARNING: Test DB initialization failed: {e}. Tests requiring DB will likely fail.")

    @classmethod
    def tearDownClass(cls):
        """
        Cleans up the test environment after all tests in the class have run.

        This method performs the following cleanup actions:
        - Removes the test-specific HDF5 and ANN index directories and their contents.
        - Restores the original HDF5 and ANN index directory paths in the
          `tensorus.client` module.
        - Restores the original database connection parameters in the
          `tensorus.client` module.
        - Notes where test database cleanup (e.g., dropping tables) would occur
          in a full-fledged test environment.
        """
        # Clean up test directories
        if os.path.exists(HDF5_DIR_TEST):
            shutil.rmtree(HDF5_DIR_TEST)
        if os.path.exists(ANN_INDEX_DIR_TEST):
            shutil.rmtree(ANN_INDEX_DIR_TEST)

        # Restore original paths and DB config
        client.HDF5_DIR = cls.original_hdf5_dir
        client.ANN_INDEX_DIR = cls.original_ann_index_dir
        client.DB_HOST = cls.original_db_host
        client.DB_NAME = cls.original_db_name
        client.DB_USER = cls.original_db_user
        client.DB_PASSWORD = cls.original_db_password
        
        # NOTE: Dropping test database tables or the database itself would happen here
        # in a full test environment.
        # Example:
        # conn = client.get_db_connection() # careful, uses test DB name
        # cur = conn.cursor()
        # cur.execute("DROP TABLE IF EXISTS index_tensors, indices, tensors CASCADE;")
        # conn.commit()
        # cur.close()
        # conn.close()
        print(f"Test directories cleaned up. Test DB {DB_NAME_TEST} would be cleaned here.")


    def setUp(self):
        # Individual test setup if needed - e.g., creating specific tensors/indices
        # For now, we assume setUpClass handles most global setup.
        # We might need to clean specific DB tables between tests if they are not independent.
        # For simplicity, the main example script cleans up files at the start.
        # Here, we rely on tearDownClass for major cleanup.
        pass

    def tearDown(self):
        # Clean up specific items created in a test, if necessary.
        # For example, if a test creates specific tensors/indices by ID, delete them.
        # This is important if tests are not perfectly isolated by setUpClass/tearDownClass.
        # For now, we'll keep it simple.
        # Example: if self.test_tensor_id: client.delete_tensor(self.test_tensor_id)
        pass

    def test_01_create_and_get_tensor(self):
        """
        Tests the creation of a tensor and its subsequent retrieval and verification.

        It performs the following steps:
        1. Creates a new tensor with sample data and metadata.
        2. Asserts that a tensor ID is returned.
        3. Retrieves the tensor using the returned ID.
        4. Verifies that the retrieved tensor data and metadata match the originals.
        5. Cleans up by deleting the created tensor.

        Note: This test requires a functional database connection.
        """
        # This test depends on a running PostgreSQL database.
        print("\nRunning test_01_create_and_get_tensor...")
        try:
            tensor_data = np.random.rand(1, 128).astype('float32')
            metadata = {"test_name": "tensor_for_test_01", "value": 123}
            
            tensor_id = client.create_tensor(tensor_data, metadata)
            self.assertIsNotNone(tensor_id)
            print(f"  Created tensor with ID: {tensor_id}")

            ret_tensor_data, ret_metadata = client.get_tensor(str(tensor_id))
            self.assertTrue(np.array_equal(tensor_data, ret_tensor_data))
            self.assertEqual(metadata, ret_metadata)
            print(f"  Successfully retrieved and verified tensor {tensor_id}")

            # Clean up this specific tensor
            client.delete_tensor(str(tensor_id))
            print(f"  Cleaned up tensor {tensor_id}")

        except Exception as e:
            print(f"  test_01_create_and_get_tensor FAILED with error: {e}")
            self.fail(f"Test failed due to database or other operational error: {e}")
            
    def test_02_create_index(self):
        """
        Tests the creation of an index and the verification of its details.

        It performs the following steps:
        1. Creates a new index with a specified name, type, dimension, and metric.
        2. Asserts that an index ID is returned.
        3. Retrieves the index details using the returned ID.
        4. Verifies that the retrieved details match the parameters used for creation.
        5. Cleans up by deleting the created index.

        Note: This test requires a functional database connection.
        """
        # This test depends on a running PostgreSQL database.
        print("\nRunning test_02_create_index...")
        try:
            index_name = "test_index_creation"
            index_type = "FLAT" # Using a simple type
            dimension = 128
            metric_type = "L2" # Faiss specific, or 'euclidean'
            
            index_id = client.create_index(index_name, index_type, dimension, metric_type)
            self.assertIsNotNone(index_id)
            print(f"  Created index with ID: {index_id}, Name: {index_name}")

            details = client.get_index_details(str(index_id))
            self.assertEqual(details[0], index_name) # name
            self.assertEqual(details[1].upper(), index_type.upper()) # type
            self.assertEqual(details[2], dimension) # dim
            self.assertEqual(details[3].lower(), metric_type.lower()) # metric
            print(f"  Successfully retrieved and verified index {index_id}")

            # Clean up this specific index
            client.delete_index(str(index_id))
            print(f"  Cleaned up index {index_id}")

        except Exception as e:
            print(f"  test_02_create_index FAILED with error: {e}")
            self.fail(f"Test failed due to database or other operational error: {e}")

    # Add more tests here, adapting from the if __name__ == "__main__": block
    # For example:
    # test_update_tensor
    # test_delete_tensor
    # test_add_tensor_to_index
    # test_build_index (will be complex due to data requirements)
    # test_query_index (also complex)
    # test_find_tensors_by_metadata
    # test_agentic_query (with mocked/simulated NLU)

if __name__ == '__main__':
    # Create the 'tensorus' directory if it doesn't exist at the root
    # to allow 'from tensorus import client' to work when run from tests/
    if not os.path.exists("../tensorus"):
        print("Warning: tensorus package directory not found at ../tensorus. Create it for imports to work if needed.")

    unittest.main()
