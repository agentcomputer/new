import h5py
import numpy as np
import uuid
import json
import psycopg2 # Or sqlalchemy
import faiss # Or annoy, nmslib, pgvector integration
import transformers # Or other LLM library
import os
import datetime
import logging

# --- Configuration ---
DB_HOST = "localhost"
DB_NAME = "tensorus_db"
DB_USER = "tensorus_user"
DB_PASSWORD = "tensorus_password"
HDF5_DIR = "./tensorus_tensors"
ANN_INDEX_DIR = "./tensorus_indices" # For Faiss/Annoy

logging.basicConfig(level=logging.INFO)

# --- Database Connection ---
def get_db_connection():
    """
    Establishes and returns a connection to the PostgreSQL database.

    Uses connection parameters (DB_HOST, DB_NAME, DB_USER, DB_PASSWORD)
    defined globally in the module.

    Returns:
        psycopg2.extensions.connection: A connection object to the PostgreSQL database.

    Raises:
        Exception: If any error occurs during the connection attempt (e.g.,
                   psycopg2.OperationalError for connection failure).
    """
    try:
        conn = psycopg2.connect(
            host=DB_HOST,
            database=DB_NAME,
            user=DB_USER,
            password=DB_PASSWORD
        )
        return conn
    except Exception as e:
        logging.error(f"Database connection error: {e}")
        raise

# --- Database Schema Initialization ---
def init_db():
    """
    Initializes the database schema by creating necessary tables if they don't exist.

    Tables created:
        - tensors: Stores metadata about each tensor.
        - indices: Stores information about ANN indices.
        - index_tensors: Maps tensors to indices.

    The function uses the global database connection parameters.
    It commits the changes if successful, or rolls back on error.

    Raises:
        Exception: If any error occurs during schema initialization (e.g.,
                   database connection issues, SQL errors).
    """
    conn = get_db_connection()
    cur = conn.cursor()
    try:
        cur.execute("""
            CREATE TABLE IF NOT EXISTS tensors (
                tensor_id UUID PRIMARY KEY,
                hdf5_filepath VARCHAR NOT NULL,
                hdf5_dataset_name VARCHAR NOT NULL,
                shape JSONB NOT NULL,
                dtype VARCHAR NOT NULL,
                size_bytes BIGINT NOT NULL,
                created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
                updated_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
                metadata_json JSONB DEFAULT '{}'
            );
        """)
        cur.execute("""
            CREATE TABLE IF NOT EXISTS indices (
                index_id UUID PRIMARY KEY,
                index_name VARCHAR UNIQUE NOT NULL,
                index_type VARCHAR NOT NULL,
                dimension INTEGER NOT NULL,
                metric_type VARCHAR NOT NULL,
                config_json JSONB DEFAULT '{}',
                status VARCHAR DEFAULT 'created', -- e.g., created, building, ready, error
                created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
                updated_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
            );
        """)
        cur.execute("""
            CREATE TABLE IF NOT EXISTS index_tensors (
                index_id UUID REFERENCES indices(index_id) ON DELETE CASCADE,
                tensor_id UUID REFERENCES tensors(tensor_id) ON DELETE CASCADE,
                position INTEGER, -- Optional: for ordered indices or specific ANN libraries
                PRIMARY KEY (index_id, tensor_id)
            );
        """)
        conn.commit()
        logging.info("Database schema initialized successfully.")
    except Exception as e:
        conn.rollback()
        logging.error(f"Error initializing database schema: {e}")
        raise
    finally:
        cur.close()
        conn.close()

# --- HDF5 Helper Functions ---
def get_hdf5_path(tensor_id):
    """
    Generates the HDF5 file path for a given tensor ID.

    Ensures the HDF5_DIR directory exists. The current strategy is one HDF5
    file per tensor, named using the tensor's UUID.

    Args:
        tensor_id (uuid.UUID or str): The UUID of the tensor.

    Returns:
        str: The absolute or relative file path for the HDF5 file.
    """
    # Simple strategy: one file per tensor
    os.makedirs(HDF5_DIR, exist_ok=True)
    return os.path.join(HDF5_DIR, f"{tensor_id}.h5")

def save_tensor_to_hdf5(tensor_id, tensor_data, metadata):
    """
    Saves tensor data and its metadata to an HDF5 file.

    The tensor data is saved as a dataset within the HDF5 file.
    Essential metadata (shape, dtype, timestamps, and user-provided metadata)
    are stored as attributes of this dataset. The HDF5 file is named after the
    tensor_id and the dataset within the file is also named after the tensor_id.

    Args:
        tensor_id (uuid.UUID or str): The UUID of the tensor, used for filename and dataset name.
        tensor_data (np.ndarray): The numpy array containing the tensor data.
        metadata (dict, optional): User-defined metadata to store alongside the tensor.
                                   This will be JSON serialized.

    Raises:
        Exception: If any error occurs during HDF5 file creation or writing (e.g.,
                   IOError, h5py errors).
    """
    filepath = get_hdf5_path(tensor_id)
    dataset_name = str(tensor_id) # Use UUID as dataset name
    try:
        with h5py.File(filepath, 'w') as f: # 'w' creates new file or overwrites
            dataset = f.create_dataset(dataset_name, data=tensor_data)
            # Store essential metadata directly as HDF5 attributes
            dataset.attrs['shape'] = list(tensor_data.shape) # h5py prefers lists for shapes
            dataset.attrs['dtype'] = str(tensor_data.dtype)
            dataset.attrs['created_at'] = datetime.datetime.now().isoformat()
            dataset.attrs['updated_at'] = datetime.datetime.now().isoformat()
            dataset.attrs['metadata_json'] = json.dumps(metadata or {})
        logging.info(f"Tensor {tensor_id} saved to {filepath} in dataset {dataset_name}")
    except Exception as e:
        logging.error(f"Error saving tensor {tensor_id} to HDF5: {e}")
        raise

def load_tensor_from_hdf5(tensor_id):
    """
    Loads tensor data and its attributes from an HDF5 file.

    Retrieves the tensor data and metadata (shape, dtype, user-defined JSON metadata)
    stored as attributes within the HDF5 dataset.

    Args:
        tensor_id (uuid.UUID or str): The UUID of the tensor to load. The HDF5 file
                                      and dataset within it are named by this ID.

    Returns:
        tuple: A tuple containing:
            - np.ndarray: The loaded tensor data.
            - tuple: The shape of the tensor.
            - str: The data type of the tensor (e.g., 'float32').
            - dict: The user-defined metadata retrieved from HDF5 attributes.

    Raises:
        FileNotFoundError: If the HDF5 file for the tensor_id does not exist.
        KeyError: If the dataset corresponding to tensor_id is not found in the HDF5 file.
        Exception: If any other error occurs during HDF5 file reading or attribute access.
    """
    # Corresponds to get_tensor in overall plan
    filepath = get_hdf5_path(tensor_id) # Assumes one file per tensor
    dataset_name = str(tensor_id)
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"HDF5 file not found for tensor {tensor_id} at {filepath}")
    try:
        with h5py.File(filepath, 'r') as f:
            if dataset_name not in f:
                raise KeyError(f"Dataset {dataset_name} not found in HDF5 file {filepath}")
            dataset = f[dataset_name]
            tensor_data = dataset[:]
            # Retrieve metadata from HDF5 attributes
            shape = tuple(dataset.attrs['shape'])
            dtype = dataset.attrs['dtype']
            h5_metadata = json.loads(dataset.attrs.get('metadata_json', '{}'))
            # created_at and updated_at are also available if needed
        logging.info(f"Tensor {tensor_id} loaded from {filepath}, dataset {dataset_name}")
        return tensor_data, shape, dtype, h5_metadata
    except Exception as e:
        logging.error(f"Error loading tensor {tensor_id} from HDF5: {e}")
        raise

def delete_tensor_from_hdf5(tensor_id):
    """
    Deletes the HDF5 file associated with a given tensor ID.

    This function implements a strategy where each tensor has its own HDF5 file.
    Deleting the tensor from HDF5 storage means removing this file.

    Args:
        tensor_id (uuid.UUID or str): The UUID of the tensor whose HDF5 file is to be deleted.

    Raises:
        Exception: If any error occurs during file deletion (e.g., OSError).
                   Does not raise an error if the file is already not found, only logs a warning.
    """
    filepath = get_hdf5_path(tensor_id)
    # For the one-file-per-tensor strategy, deleting the dataset means deleting the file.
    if os.path.exists(filepath):
        try:
            os.remove(filepath)
            logging.info(f"HDF5 file {filepath} for tensor {tensor_id} deleted.")
        except Exception as e:
            logging.error(f"Error deleting HDF5 file {filepath} for tensor {tensor_id}: {e}")
            raise
    else:
        logging.warning(f"HDF5 file not found for deletion: {filepath}")


# --- Tensorus API Functions ---
def create_tensor(tensor_data, metadata=None):
    """
    Creates a new tensor, saves its data to HDF5, and records its metadata in PostgreSQL.

    A unique UUID is generated for the tensor. The tensor data is converted to a NumPy
    array, and its properties (shape, dtype, size) are recorded. The data is stored
    in an HDF5 file, and its metadata (including the HDF5 path and user-provided
    metadata) is inserted into the 'tensors' table in the database.

    Args:
        tensor_data (array-like): The data for the tensor (e.g., list, np.ndarray).
                                  Will be converted to np.ndarray.
        metadata (dict, optional): User-defined metadata to associate with the tensor.
                                   Defaults to None, stored as an empty JSON object if so.

    Returns:
        uuid.UUID: The unique ID generated for the created tensor.

    Raises:
        Exception: If HDF5 saving fails or if the database insertion fails.
                   Attempts to clean up the HDF5 file if DB insertion fails.
    """
    tensor_id = uuid.uuid4()
    tensor_data_np = np.asarray(tensor_data) # Ensure numpy array

    shape = list(tensor_data_np.shape) # Use list for JSON serialization
    dtype = str(tensor_data_np.dtype)
    size_bytes = tensor_data_np.nbytes
    hdf5_filepath = get_hdf5_path(tensor_id) # Path for the new HDF5 file
    hdf5_dataset_name = str(tensor_id) # Dataset name is the tensor_id

    # 1. Save to HDF5
    # The metadata passed to HDF5 might be a subset or all of it.
    # The design doc mentions: shape, dtype, created_at, updated_at, metadata_json as HDF5 attributes.
    save_tensor_to_hdf5(tensor_id, tensor_data_np, metadata)

    # 2. Insert into PostgreSQL
    conn = get_db_connection()
    cur = conn.cursor()
    try:
        cur.execute(
            "INSERT INTO tensors (tensor_id, hdf5_filepath, hdf5_dataset_name, shape, dtype, size_bytes, metadata_json) "
            "VALUES (%s, %s, %s, %s, %s, %s, %s);",
            (str(tensor_id), hdf5_filepath, hdf5_dataset_name, json.dumps(shape), dtype, int(size_bytes), json.dumps(metadata or {}))
        )
        conn.commit()
        logging.info(f"Tensor {tensor_id} metadata created in database.")
        return tensor_id
    except Exception as e:
        conn.rollback()
        logging.error(f"Error creating tensor {tensor_id} metadata in database: {e}")
        # Clean up HDF5 file if DB insert fails
        try:
            delete_tensor_from_hdf5(tensor_id)
        except Exception as cleanup_e:
            logging.error(f"Error during HDF5 cleanup for failed tensor {tensor_id} creation: {cleanup_e}")
        raise
    finally:
        cur.close()
        conn.close()

def get_tensor(tensor_id_str):
    """
    Retrieves a tensor's data and its primary user-defined metadata.

    The function fetches metadata from the PostgreSQL 'tensors' table and loads
    the tensor data from the corresponding HDF5 file.

    Args:
        tensor_id_str (str or uuid.UUID): The UUID of the tensor to retrieve.
                                          If str, it's converted to UUID.

    Returns:
        tuple: A tuple containing:
            - np.ndarray: The tensor data loaded from HDF5.
            - dict: The user-defined metadata associated with the tensor (from PostgreSQL).

    Raises:
        KeyError: If the tensor_id is not found in the database.
        FileNotFoundError: If the HDF5 file for the tensor is not found (raised by
                           load_tensor_from_hdf5).
        Exception: Can also propagate exceptions from database connection/query
                   or HDF5 loading.
    """
    # Matches API 3.1
    tensor_id = uuid.UUID(tensor_id_str) if isinstance(tensor_id_str, str) else tensor_id_str
    conn = get_db_connection()
    cur = conn.cursor()
    db_metadata = None
    try:
        cur.execute(
            "SELECT metadata_json FROM tensors WHERE tensor_id = %s;",
            (str(tensor_id),)
        )
        result = cur.fetchone()
        if not result:
            raise KeyError(f"Tensor {tensor_id} not found in database.")
        db_metadata = json.loads(result[0]) if result[0] else {}
    except Exception as e:
        logging.error(f"Error getting tensor {tensor_id} metadata from database: {e}")
        raise
    finally:
        cur.close()
        conn.close()

    # Load tensor data and HDF5 attributes
    # The load_tensor_from_hdf5 function already returns tensor_data and h5_metadata
    tensor_data, _, _, _ = load_tensor_from_hdf5(tensor_id) # shape, dtype, h5_metadata also returned but primary need is tensor_data and DB metadata

    # The design asks for tensor_data and metadata.
    # The metadata in DB (metadata_json) is the primary source for user-defined metadata.
    return tensor_data, db_metadata


def update_tensor(tensor_id_str, tensor_data=None, metadata=None):
    """
    Updates a tensor's data and/or its metadata.

    If tensor_data is provided, the HDF5 file for the tensor is overwritten with
    the new data, and its HDF5 attributes (shape, dtype, size, metadata_json) are updated.
    The corresponding database record (shape, dtype, size_bytes, updated_at) is also updated.

    If metadata is provided, the 'metadata_json' field in the PostgreSQL 'tensors'
    table is updated. If tensor_data was not also provided (meaning HDF5 was not
    re-saved), this function will also update the 'metadata_json' attribute in the
    HDF5 file by reloading the existing data and re-saving it with the new metadata.

    Args:
        tensor_id_str (str or uuid.UUID): The UUID of the tensor to update.
        tensor_data (array-like, optional): New data for the tensor. If None,
                                            data is not updated. Defaults to None.
        metadata (dict, optional): New metadata to merge with existing metadata.
                                   If None, metadata is not updated. Defaults to None.

    Raises:
        KeyError: If the tensor_id is not found in the database.
        Exception: Propagates exceptions from database operations or HDF5 operations.
                   Rolls back database changes on error.
    """
    # Matches API 3.1
    tensor_id = uuid.UUID(tensor_id_str) if isinstance(tensor_id_str, str) else tensor_id_str
    conn = get_db_connection()
    cur = conn.cursor()
    try:
        # Check if tensor exists and get its current hdf5_filepath and metadata
        cur.execute("SELECT hdf5_filepath, metadata_json FROM tensors WHERE tensor_id = %s;", (str(tensor_id),))
        result = cur.fetchone()
        if not result:
            raise KeyError(f"Tensor {tensor_id} not found for update.")
        
        current_hdf5_filepath, current_metadata_json_str = result
        current_metadata = json.loads(current_metadata_json_str or '{}')

        if tensor_data is not None:
            tensor_data_np = np.asarray(tensor_data)
            # Overwrite the existing HDF5 file or dataset for this tensor_id
            # save_tensor_to_hdf5 handles HDF5 part including updating attributes
            # We need to pass the *updated* metadata to save_tensor_to_hdf5 if metadata is also changing
            # or the existing metadata if only tensor_data changes.
            
            metadata_for_hdf5 = current_metadata.copy()
            if metadata is not None:
                metadata_for_hdf5.update(metadata)

            save_tensor_to_hdf5(tensor_id, tensor_data_np, metadata_for_hdf5) # This updates HDF5 attrs
            
            # Update shape and size_bytes in DB if tensor_data changed
            new_shape = list(tensor_data_np.shape)
            new_dtype = str(tensor_data_np.dtype)
            new_size_bytes = tensor_data_np.nbytes
            cur.execute(
                "UPDATE tensors "
                "SET shape = %s, dtype = %s, size_bytes = %s, updated_at = CURRENT_TIMESTAMP "
                "WHERE tensor_id = %s;",
                (json.dumps(new_shape), new_dtype, int(new_size_bytes), str(tensor_id))
            )
            logging.info(f"Tensor {tensor_id} data updated in HDF5 and database (shape, dtype, size).")

        if metadata is not None:
            # Update metadata_json in the database
            # The HDF5 metadata_json attribute is handled by save_tensor_to_hdf5 if tensor_data was also provided.
            # If only metadata is updated, we need to specifically update HDF5 attributes.
            updated_db_metadata = current_metadata.copy()
            updated_db_metadata.update(metadata)

            if tensor_data is None: # If tensor data was not updated, HDF5 attributes need separate update
                # Reload tensor data to re-save with new metadata, or update attrs directly
                # For simplicity, re-save. This assumes tensor_data is unchanged.
                # A more optimized way would be to open HDF5 and update only attributes.
                temp_tensor_data, _, _, _ = load_tensor_from_hdf5(tensor_id)
                save_tensor_to_hdf5(tensor_id, temp_tensor_data, updated_db_metadata)

            cur.execute(
                "UPDATE tensors "
                "SET metadata_json = %s, updated_at = CURRENT_TIMESTAMP "
                "WHERE tensor_id = %s;",
                (json.dumps(updated_db_metadata), str(tensor_id))
            )
            logging.info(f"Tensor {tensor_id} metadata updated in database.")
        
        conn.commit()

    except Exception as e:
        conn.rollback()
        logging.error(f"Error updating tensor {tensor_id}: {e}")
        raise
    finally:
        cur.close()
        conn.close()

def delete_tensor(tensor_id_str):
    """
    Deletes a tensor from the system.

    This involves:
    1. Deleting the associated HDF5 file.
    2. Deleting any references to this tensor from the 'index_tensors' table.
    3. Deleting the tensor's metadata record from the 'tensors' table.

    Args:
        tensor_id_str (str or uuid.UUID): The UUID of the tensor to delete.

    Raises:
        Exception: Propagates exceptions from database operations or HDF5 file deletion.
                   Rolls back database changes on error.
                   Does not raise error if tensor not found in DB, only logs warning.
    """
    # Matches API 3.1
    tensor_id = uuid.UUID(tensor_id_str) if isinstance(tensor_id_str, str) else tensor_id_str
    conn = get_db_connection()
    cur = conn.cursor()
    try:
        # First, verify tensor exists to get HDF5 path for cleanup
        cur.execute("SELECT hdf5_filepath FROM tensors WHERE tensor_id = %s;", (str(tensor_id),))
        result = cur.fetchone()
        if not result:
            logging.warning(f"Tensor {tensor_id} not found in database for deletion.")
            # Depending on desired behavior, could raise KeyError or just return
            return

        # 1. Delete from HDF5
        delete_tensor_from_hdf5(tensor_id) # Uses the get_hdf5_path strategy

        # 2. Delete from index_tensors (CASCADE should handle this, but explicit is safer for awareness)
        cur.execute("DELETE FROM index_tensors WHERE tensor_id = %s;", (str(tensor_id),))
        
        # 3. Delete from tensors table
        cur.execute("DELETE FROM tensors WHERE tensor_id = %s;", (str(tensor_id),))
        
        conn.commit()
        logging.info(f"Tensor {tensor_id} and its references deleted from database and HDF5.")

    except Exception as e:
        conn.rollback()
        logging.error(f"Error deleting tensor {tensor_id}: {e}")
        raise
    finally:
        cur.close()
        conn.close()

# --- Indexing API Functions ---
def get_index_path(index_id):
    """
    Generates the file path for an ANN index file (e.g., Faiss index).

    Ensures the ANN_INDEX_DIR directory exists. The current strategy is one
    file per index, named using the index's UUID and a '.faiss' extension.

    Args:
        index_id (uuid.UUID or str): The UUID of the index.

    Returns:
        str: The file path for the ANN index.
    """
    os.makedirs(ANN_INDEX_DIR, exist_ok=True)
    return os.path.join(ANN_INDEX_DIR, f"{index_id}.faiss") # Example for Faiss

def create_index(index_name, index_type, dimension, metric_type, config=None):
    """
    Creates a new ANN index record in the database.

    This function records the metadata for an index, such as its name, type (e.g., HNSW, IVFFlat),
    vector dimension, metric type (e.g., L2, cosine), and any specific configuration.
    The actual ANN index file (e.g., Faiss index) is typically created and populated
    during the `build_index` step.

    Args:
        index_name (str): A unique name for the index.
        index_type (str): The type of ANN index (e.g., 'HNSW', 'IVFFLAT', 'FLAT').
        dimension (int): The dimensionality of the vectors that will be stored in this index.
        metric_type (str): The distance metric to be used (e.g., 'L2', 'cosine', 'IP' for inner product).
        config (dict, optional): A dictionary for additional configuration parameters specific
                                 to the index type (e.g., M and ef_construction for HNSW).
                                 Defaults to None, stored as an empty JSON object.

    Returns:
        uuid.UUID: The unique ID generated for the created index.

    Raises:
        Exception: Propagates exceptions from database operations (e.g., if index_name is not unique).
                   Rolls back database changes on error.
    """
    # API 3.2
    index_id = uuid.uuid4()
    config = config or {}
    conn = get_db_connection()
    cur = conn.cursor()
    try:
        cur.execute(
            "INSERT INTO indices (index_id, index_name, index_type, dimension, metric_type, config_json, status) "
            "VALUES (%s, %s, %s, %s, %s, %s, %s);",
            (str(index_id), index_name, index_type, dimension, metric_type.lower(), json.dumps(config), 'created')
        )
        conn.commit()
        logging.info(f"Index {index_name} (ID: {index_id}) created in database with type {index_type}, dim {dimension}, metric {metric_type}.")
        # Actual index file creation/initialization happens at build_index time for Faiss
        return index_id
    except Exception as e:
        conn.rollback()
        logging.error(f"Error creating index {index_name}: {e}")
        raise
    finally:
        cur.close()
        conn.close()

def get_index_details(index_id_str):
    """
    Retrieves detailed information about a specific ANN index from the database.

    Args:
        index_id_str (str or uuid.UUID): The UUID of the index to retrieve details for.

    Returns:
        tuple: A tuple containing:
            - str: The name of the index.
            - str: The type of the index (e.g., 'HNSW').
            - int: The dimension of vectors in the index.
            - str: The metric type used by the index (e.g., 'l2', 'cosine').
            - dict: The configuration JSON stored for the index.
            - str: The current status of the index (e.g., 'created', 'building', 'ready', 'error').

    Raises:
        KeyError: If the index_id is not found in the database.
        Exception: Propagates exceptions from database operations.
    """
    # Helper
    index_id = uuid.UUID(index_id_str) if isinstance(index_id_str, str) else index_id_str
    conn = get_db_connection()
    cur = conn.cursor()
    try:
        cur.execute(
            "SELECT index_name, index_type, dimension, metric_type, config_json, status "
            "FROM indices "
            "WHERE index_id = %s;",
            (str(index_id),)
        )
        result = cur.fetchone()
        if not result:
            raise KeyError(f"Index {index_id} not found.")
        name, type, dim, metric, cfg_json, status = result
        return name, type, dim, metric, json.loads(cfg_json), status
    except Exception as e:
        logging.error(f"Error getting details for index {index_id}: {e}")
        raise
    finally:
        cur.close()
        conn.close()

def add_tensor_to_index(index_id_str, tensor_id_str, position=None):
    """
    Associates a tensor with an index in the 'index_tensors' tracking table.

    This function creates a record indicating that a specific tensor should be
    included in a specific index. The actual addition of the tensor data to the
    ANN index file happens during the `build_index` process.

    Args:
        index_id_str (str or uuid.UUID): The UUID of the index.
        tensor_id_str (str or uuid.UUID): The UUID of the tensor.
        position (int, optional): An optional integer indicating a specific position or
                                  ID for the tensor within the index, if supported by the
                                  ANN library and index type. Defaults to None.

    Raises:
        Exception: Propagates exceptions from database operations (e.g., foreign key
                   violations if index_id or tensor_id do not exist).
                   Rolls back database changes on error.
    """
    # API 3.2
    index_id = uuid.UUID(index_id_str) if isinstance(index_id_str, str) else index_id_str
    tensor_id = uuid.UUID(tensor_id_str) if isinstance(tensor_id_str, str) else tensor_id_str
    conn = get_db_connection()
    cur = conn.cursor()
    try:
        cur.execute(
            "INSERT INTO index_tensors (index_id, tensor_id, position) "
            "VALUES (%s, %s, %s);",
            (str(index_id), str(tensor_id), position)
        )
        conn.commit()
        logging.info(f"Tensor {tensor_id} added to index {index_id} tracking table.")
    except Exception as e:
        conn.rollback()
        logging.error(f"Error adding tensor {tensor_id} to index {index_id} tracking: {e}")
        raise
    finally:
        cur.close()
        conn.close()

def build_index(index_id_str):
    """
    Builds (or rebuilds) an ANN index file (e.g., Faiss index).

    This function loads all tensors associated with the given index_id from the
    'index_tensors' table, retrieves their data from HDF5, and then constructs
    the specified type of Faiss index (e.g., HNSW, IVFFlat, FlatL2, FlatIP).
    The built index is then saved to a file.
    The status of the index in the 'indices' table is updated during and after the build.

    Args:
        index_id_str (str or uuid.UUID): The UUID of the index to build.

    Raises:
        ValueError: If no tensors are found for the index, if no valid tensor data
                    could be loaded, or if an unsupported index_type or metric_type
                    is specified.
        Exception: Propagates exceptions from database operations, HDF5 loading,
                   or Faiss index building/saving. Updates index status to 'error'
                   on failure.
    """
    # API 3.2
    index_id = uuid.UUID(index_id_str) if isinstance(index_id_str, str) else index_id_str
    
    name, index_type, dimension, metric_type, config, status = get_index_details(index_id)
    
    if status == 'building':
        logging.warning(f"Index {index_id} is already being built. Skipping.")
        return
    if status == 'ready' and not config.get('allow_rebuild', False):
        logging.info(f"Index {index_id} is already ready. Skipping build. Set 'allow_rebuild':true in config to force.")
        return

    conn = get_db_connection()
    cur = conn.cursor()
    try:
        cur.execute("UPDATE indices SET status = 'building', updated_at = CURRENT_TIMESTAMP WHERE index_id = %s;", (str(index_id),))
        conn.commit()

        cur.execute(
            "SELECT t.tensor_id, t.hdf5_filepath, t.hdf5_dataset_name "
            "FROM tensors t "
            "JOIN index_tensors it ON t.tensor_id = it.tensor_id "
            "WHERE it.index_id = %s;",
            (str(index_id),)
        )
        tensor_records = cur.fetchall()

        if not tensor_records:
            logging.warning(f"No tensors found for index {index_id}. Index will be empty.")
            if index_type.lower() in ['hnsw', 'ivfflat', 'flatl2', 'flatip']:
                idx_faiss = None
                if metric_type.lower() == 'euclidean' or metric_type.lower() == 'l2':
                    idx_faiss = faiss.IndexFlatL2(dimension)
                elif metric_type.lower() == 'cosine' or metric_type.lower() == 'ip':
                    idx_faiss = faiss.IndexFlatIP(dimension)
                else:
                    raise ValueError(f"Unsupported metric type for empty Faiss index: {metric_type}")
                
                if index_type.lower() == 'hnsw':
                    m_val = config.get('M', 32)
                    ef_cons = config.get('ef_construction', 200)
                    hnsw_index = faiss.IndexHNSWFlat(idx_faiss, m_val)
                    hnsw_index.hnsw.efConstruction = ef_cons
                    idx_faiss = hnsw_index
                elif index_type.lower() == 'ivfflat':
                    nlist = config.get('nlist', 100)
                    quantizer = idx_faiss
                    ivf_index = faiss.IndexIVFFlat(quantizer, dimension, nlist)
                    idx_faiss = ivf_index
                
                if idx_faiss:
                    index_filepath = get_index_path(index_id)
                    faiss.write_index(idx_faiss, index_filepath)
                    logging.info(f"Empty Faiss index {index_id} of type {index_type} created as no tensors were found.")
            
            cur.execute("UPDATE indices SET status = 'ready', updated_at = CURRENT_TIMESTAMP WHERE index_id = %s;", (str(index_id),))
            conn.commit()
            return

        all_tensor_data = []
        tensor_ids_in_order = []
        
        logging.info(f"Building index {index_id} ({name}) with {len(tensor_records)} tensors...")
        for record_idx, record in enumerate(tensor_records):
            t_id, h5_path, dset_name = record
            try:
                tensor_data, _, _, _ = load_tensor_from_hdf5(t_id)
                if tensor_data.ndim > 2:
                    logging.warning(f"Tensor {t_id} has unsupported ndim {tensor_data.ndim}. Skipping.")
                    continue
                current_dim = tensor_data.shape[-1]
                if tensor_data.ndim == 1 and tensor_data.shape[0] != dimension:
                    logging.warning(f"Tensor {t_id} dim {tensor_data.shape[0]} !~ index dim {dimension}. Skipping.")
                    continue
                if tensor_data.ndim == 2 and tensor_data.shape[1] != dimension:
                    if tensor_data.shape[0] == 1:
                        tensor_data = tensor_data.reshape(dimension)
                    else:
                        logging.warning(f"Tensor {t_id} is a batch of vectors. Taking first vector. Shape: {tensor_data.shape}")
                        tensor_data = tensor_data[0]
                        if tensor_data.shape[0] != dimension:
                             logging.warning(f"First vector of Tensor {t_id} dim {tensor_data.shape[0]} !~ index dim {dimension}. Skipping.")
                             continue
                all_tensor_data.append(tensor_data.astype('float32'))
                tensor_ids_in_order.append(t_id)
            except Exception as e:
                logging.error(f"Failed to load or process tensor {t_id} for index build: {e}. Skipping.")
                continue
        
        if not all_tensor_data:
            cur.execute("UPDATE indices SET status = 'error', updated_at = CURRENT_TIMESTAMP WHERE index_id = %s;", (str(index_id),))
            conn.commit()
            raise ValueError(f"No valid tensor data could be loaded to build index {index_id}.")

        data_array = np.array(all_tensor_data)
        if data_array.ndim == 1 :
             data_array = data_array.reshape(1, -1)

        faiss_index = None
        final_metric_type = None

        if metric_type.lower() == 'euclidean' or metric_type.lower() == 'l2':
            final_metric_type = faiss.METRIC_L2
        elif metric_type.lower() == 'cosine' or metric_type.lower() == 'ip':
            faiss.normalize_L2(data_array)
            final_metric_type = faiss.METRIC_INNER_PRODUCT
        else:
            cur.execute("UPDATE indices SET status = 'error', updated_at = CURRENT_TIMESTAMP WHERE index_id = %s;", (str(index_id),))
            conn.commit()
            raise ValueError(f"Unsupported metric_type for Faiss: {metric_type}")

        if index_type.lower() == 'flatl2' or (index_type.lower() == 'flat' and final_metric_type == faiss.METRIC_L2):
            faiss_index = faiss.IndexFlatL2(dimension)
        elif index_type.lower() == 'flatip' or (index_type.lower() == 'flat' and final_metric_type == faiss.METRIC_INNER_PRODUCT):
            faiss_index = faiss.IndexFlatIP(dimension)
        elif index_type.lower() == 'hnsw':
            m_val = config.get('M', 32)
            ef_cons = config.get('ef_construction', 200)
            base_idx = faiss.IndexFlat(dimension, final_metric_type)
            faiss_index = faiss.IndexHNSWFlat(base_idx, m_val)
            faiss_index.hnsw.efConstruction = ef_cons
        elif index_type.lower() == 'ivfflat':
            nlist = config.get('nlist', max(1, min(len(data_array) // 40, 1000)))
            quantizer = faiss.IndexFlat(dimension, final_metric_type)
            faiss_index = faiss.IndexIVFFlat(quantizer, dimension, nlist, final_metric_type)
            if not faiss_index.is_trained and len(data_array) > 0:
                 faiss_index.train(data_array)
            else:
                 logging.warning(f"IVFFlat index for {index_id} not trained as there is no data or it's already marked trained.")
        else:
            cur.execute("UPDATE indices SET status = 'error', updated_at = CURRENT_TIMESTAMP WHERE index_id = %s;", (str(index_id),))
            conn.commit()
            raise ValueError(f"Unsupported index_type: {index_type}")
        
        id_map_index = faiss.IndexIDMap(faiss_index)
        faiss_ids_for_index = np.arange(len(tensor_ids_in_order)).astype('int64')
        id_map_index.add_with_ids(data_array, faiss_ids_for_index)
        
        index_filepath = get_index_path(index_id)
        faiss.write_index(id_map_index, index_filepath)

        for i, t_db_id in enumerate(tensor_ids_in_order):
            cur.execute(
                "UPDATE index_tensors SET position = %s WHERE index_id = %s AND tensor_id = %s;",
                (int(faiss_ids_for_index[i]), str(index_id), str(t_db_id))
            )

        cur.execute("UPDATE indices SET status = 'ready', updated_at = CURRENT_TIMESTAMP WHERE index_id = %s;", (str(index_id),))
        conn.commit()
        logging.info(f"Index {index_id} ({name}) built successfully with {id_map_index.ntotal} vectors and saved to {index_filepath}.")

    except Exception as e:
        try:
            if conn and not conn.closed and cur and not cur.closed:
                conn.rollback()
                cur.execute("UPDATE indices SET status = 'error', updated_at = CURRENT_TIMESTAMP WHERE index_id = %s;", (str(index_id),))
                conn.commit()
        except Exception as db_err:
            logging.error(f"Failed to update index {index_id} status to 'error': {db_err}")
        logging.error(f"Error building index {index_id} ({name}): {e}")
        raise
    finally:
        if cur: cur.close()
        if conn: conn.close()


def query_index(index_id_str, query_tensor_data, k, metadata_filter=None):
    """
    Queries an ANN index to find the k nearest neighbors to a given query tensor.

    Loads the specified Faiss index from its file, performs a search, and then
    maps the internal Faiss IDs back to application-level tensor_ids.
    If a metadata_filter is provided, the initial k results are further filtered
    based on their metadata in the database. The full tensor data and metadata for
    the final results are then retrieved.

    Args:
        index_id_str (str or uuid.UUID): The UUID of the index to query.
        query_tensor_data (array-like): The query tensor data (numpy array or list).
        k (int): The number of nearest neighbors to retrieve.
        metadata_filter (dict, optional): A dictionary specifying metadata conditions
                                          to filter the results. Uses PostgreSQL JSONB
                                          containment (@>) for filtering. Defaults to None.

    Returns:
        list: A list of dictionaries, where each dictionary represents a neighbor and contains:
              'tensor_id' (str), 'distance' (float), 'tensor_data' (np.ndarray),
              and 'metadata' (dict). The list is sorted by distance.
              Returns an empty list if no neighbors are found or if the index is empty.

    Raises:
        ValueError: If the index is not in a 'ready' state, or if the query tensor's
                    dimension does not match the index's dimension.
        FileNotFoundError: If the Faiss index file is not found.
        Exception: Propagates exceptions from Faiss, database, or HDF5 operations.
    """
    # API 3.2
    index_id = uuid.UUID(index_id_str) if isinstance(index_id_str, str) else index_id_str
    
    index_name, index_type, dimension, metric_type, config, status = get_index_details(index_id)

    if status != 'ready':
        raise ValueError(f"Index {index_id} ({index_name}) is not ready for querying. Status: {status}")

    index_filepath = get_index_path(index_id)
    if not os.path.exists(index_filepath):
        raise FileNotFoundError(f"Index file not found for {index_id} at {index_filepath}")

    try:
        faiss_index = faiss.read_index(index_filepath)
        if hasattr(faiss_index, 'hnsw') and 'ef_search' in config:
            faiss_index.hnsw.efSearch = config['ef_search']
        elif hasattr(faiss_index, 'nprobe') and 'nprobe' in config:
            faiss_index.nprobe = config['nprobe']
    except Exception as e:
        logging.error(f"Error loading Faiss index {index_id} from {index_filepath}: {e}")
        raise

    query_tensor_np = np.asarray(query_tensor_data).astype('float32')
    if query_tensor_np.ndim == 1:
        query_tensor_np = query_tensor_np.reshape(1, -1)
    
    if query_tensor_np.shape[1] != dimension:
        raise ValueError(f"Query tensor dimension {query_tensor_np.shape[1]} != index dimension {dimension}.")

    if metric_type.lower() == 'cosine' or metric_type.lower() == 'ip':
        faiss.normalize_L2(query_tensor_np)

    logging.info(f"Querying index {index_id} ({index_name}) with {query_tensor_np.shape[0]} vector(s) for k={k} neighbors.")
    distances, faiss_internal_ids = faiss_index.search(query_tensor_np, k=k)

    conn = get_db_connection()
    cur = conn.cursor()
    
    results = []
    valid_positions = [int(pos) for pos in faiss_internal_ids[0] if pos != -1]
    if not valid_positions:
        cur.close()
        conn.close()
        return []

    placeholders = ', '.join(['%s'] * len(valid_positions))
    cur.execute(f"""
        SELECT tensor_id, position FROM index_tensors
        WHERE index_id = %s AND position IN ({placeholders});
    """, (str(index_id),) + tuple(valid_positions))
    
    position_to_tensor_id_map = {pos: t_id for t_id, pos in cur.fetchall()}

    raw_neighbors = []
    for i, pos_idx in enumerate(faiss_internal_ids[0]):
        if pos_idx == -1 : continue
        actual_tensor_id = position_to_tensor_id_map.get(int(pos_idx))
        if actual_tensor_id:
            raw_neighbors.append({
                'tensor_id': actual_tensor_id,
                'distance': float(distances[0][i])
            })
        else:
            logging.warning(f"Could not map Faiss internal ID {pos_idx} back to a tensor_id for index {index_id}.")

    if not metadata_filter:
        final_results = []
        for neighbor in raw_neighbors:
            try:
                tensor_data, tensor_metadata = get_tensor(neighbor['tensor_id'])
                final_results.append({
                    'tensor_id': str(neighbor['tensor_id']),
                    'distance': neighbor['distance'],
                    'tensor_data': tensor_data,
                    'metadata': tensor_metadata
                })
            except Exception as e:
                logging.error(f"Error retrieving full data for tensor {neighbor['tensor_id']}: {e}")
        cur.close()
        conn.close()
        return sorted(final_results, key=lambda x: x['distance'])

    logging.info(f"Applying metadata filter: {metadata_filter}")
    # filtered_tensor_ids = [] # This variable was unused.
    final_results = [] # Initialize final_results here
    if raw_neighbors:
        neighbor_tensor_ids_to_check = [n['tensor_id'] for n in raw_neighbors]
        if not neighbor_tensor_ids_to_check:
            cur.close()
            conn.close()
            return []

        neighbor_tensor_ids_str = [str(tid) for tid in neighbor_tensor_ids_to_check]
        id_placeholders = ', '.join(['%s'] * len(neighbor_tensor_ids_str))
        
        sql_query = f"""
            SELECT tensor_id FROM tensors
            WHERE tensor_id IN ({id_placeholders}) AND metadata_json @> %s;
        """
        params = tuple(neighbor_tensor_ids_str) + (json.dumps(metadata_filter),)
        
        cur.execute(sql_query, params)
        passed_filter_records = cur.fetchall()
        passed_filter_tensor_ids = {rec[0] for rec in passed_filter_records}

        final_results_unretrieved = [
            n for n in raw_neighbors if n['tensor_id'] in passed_filter_tensor_ids
        ]
        
        # final_results = [] # Already initialized
        for neighbor in final_results_unretrieved:
            try:
                tensor_data, tensor_metadata = get_tensor(neighbor['tensor_id'])
                final_results.append({
                    'tensor_id': str(neighbor['tensor_id']),
                    'distance': neighbor['distance'],
                    'tensor_data': tensor_data,
                    'metadata': tensor_metadata
                })
            except Exception as e:
                logging.error(f"Error retrieving full data for filtered tensor {neighbor['tensor_id']}: {e}")

    cur.close()
    conn.close()
    return sorted(final_results, key=lambda x: x['distance'])


def delete_index(index_id_str):
    """
    Deletes an ANN index from the system.

    This involves:
    1. Deleting all records from 'index_tensors' that reference this index.
    2. Deleting the index's metadata record from the 'indices' table.
    3. Deleting the actual ANN index file (e.g., Faiss index file) from storage.

    Args:
        index_id_str (str or uuid.UUID): The UUID of the index to delete.

    Raises:
        Exception: Propagates exceptions from database operations or file deletion.
                   Rolls back database changes on error.
    """
    # API 3.2
    index_id = uuid.UUID(index_id_str) if isinstance(index_id_str, str) else index_id_str
    conn = get_db_connection()
    cur = conn.cursor()
    try:
        cur.execute("DELETE FROM index_tensors WHERE index_id = %s;", (str(index_id),))
        cur.execute("DELETE FROM indices WHERE index_id = %s;", (str(index_id),))
        conn.commit()
        logging.info(f"Index {index_id} records deleted from database.")

        index_filepath = get_index_path(index_id)
        if os.path.exists(index_filepath):
            os.remove(index_filepath)
            logging.info(f"Index file {index_filepath} deleted.")
        else:
            logging.warning(f"Index file {index_filepath} not found for deletion.")
            
    except Exception as e:
        conn.rollback()
        logging.error(f"Error deleting index {index_id}: {e}")
        raise
    finally:
        cur.close()
        conn.close()

# --- Agentic Interaction Functions ---
def find_tensors_by_metadata(metadata_filter, retrieve_data=False):
    """
    Finds tensors based on a metadata filter using PostgreSQL JSONB containment.

    Args:
        metadata_filter (dict): A dictionary specifying the metadata criteria.
                                Tensors whose 'metadata_json' field contains this
                                JSON structure will be returned.
        retrieve_data (bool, optional): If True, the actual tensor data will also be
                                        retrieved for each found tensor. Defaults to False.

    Returns:
        list: A list of dictionaries. Each dictionary contains:
              'tensor_id' (str), 'metadata' (dict).
              If retrieve_data is True, it also includes 'tensor_data' (np.ndarray or None).

    Raises:
        Exception: Propagates exceptions from database operations or from `get_tensor`
                   if `retrieve_data` is True.
    """
    # API 3.3
    conn = get_db_connection()
    cur = conn.cursor()
    results = []
    try:
        sql_query = "SELECT tensor_id, metadata_json FROM tensors WHERE metadata_json @> %s;"
        params = (json.dumps(metadata_filter),)
        
        cur.execute(sql_query, params)
        
        for record in cur.fetchall():
            tensor_id_uuid = record[0]
            metadata = json.loads(record[1])
            item = {'tensor_id': str(tensor_id_uuid), 'metadata': metadata}
            if retrieve_data:
                try:
                    tensor_data, _ = get_tensor(tensor_id_uuid)
                    item['tensor_data'] = tensor_data
                except Exception as e:
                    logging.error(f"Error retrieving tensor data for {tensor_id_uuid} in find_tensors_by_metadata: {e}")
                    item['tensor_data'] = None
            results.append(item)
        logging.info(f"Found {len(results)} tensors matching metadata filter: {metadata_filter}")
    except Exception as e:
        logging.error(f"Error finding tensors by metadata: {e}")
        raise
    finally:
        cur.close()
        conn.close()
    return results

def find_similar_tensors_with_metadata_filter(index_id_str, query_tensor_data, k, metadata_filter):
    """
    Finds tensors similar to a query tensor within a specified index, with an additional metadata filter.

    This is a convenience function that directly calls `query_index` with the
    provided parameters. It's designed for scenarios where similarity search
    needs to be combined with metadata-based filtering.

    Args:
        index_id_str (str or uuid.UUID): The UUID of the index to query.
        query_tensor_data (array-like): The query tensor data.
        k (int): The number of nearest neighbors to retrieve.
        metadata_filter (dict): Metadata criteria to filter the initial search results.

    Returns:
        list: A list of dictionaries, as returned by `query_index`. Each dictionary
              represents a neighbor and includes 'tensor_id', 'distance',
              'tensor_data', and 'metadata'.

    Raises:
        (Propagates exceptions from query_index)
    """
    # API 3.3
    logging.info(f"Finding similar tensors for index {index_id_str} with k={k} and filter: {metadata_filter}")
    return query_index(index_id_str, query_tensor_data, k, metadata_filter)

def agentic_query(query_string, context_data=None):
    """
    Processes a natural language query string to perform tensor-related actions.

    This function simulates a basic Natural Language Understanding (NLU) process
    to determine the user's intent (e.g., find similar tensors, find by metadata)
    and extracts relevant parameters from the query string and context_data.
    It then calls the appropriate Tensorus API functions.

    Supported intents (simplified):
    - "similar to tensor {id} [tagged with {tag}]": Triggers `find_similar_tensors_with_metadata_filter`.
    - "find tensors with metadata [user is {user_id}]": Triggers `find_tensors_by_metadata`.

    Args:
        query_string (str): The natural language query.
        context_data (dict, optional): Additional context or parameters that might
                                       not be directly in the query string.
                                       For similarity: 'k', 'metadata_filter', 'query_tensor_data'.
                                       For metadata search: 'metadata_filter', 'retrieve_data'.

    Returns:
        list or dict: The results from the executed Tensorus function, or a dictionary
                      with an 'error' key if the query is not understood, a required
                      parameter is missing, or an operation fails.

    Note:
        The NLU part is very basic and for demonstration purposes.
        It defaults to the first available 'ready' index if an index is needed but
        not specified.
    """
    # API 3.3
    logging.info(f"Received agentic query: '{query_string}' with context: {context_data}")
    intent = None
    params = {}

    conn = get_db_connection()
    cur = conn.cursor()
    cur.execute("SELECT index_id, index_name FROM indices WHERE status = 'ready';")
    available_indices = cur.fetchall()
    cur.close()
    conn.close()

    target_index_id = None
    if available_indices:
        target_index_id = str(available_indices[0][0])
        logging.info(f"Defaulting to first available ready index: {available_indices[0][1]} ({target_index_id})")

    if "similar to tensor" in query_string.lower() or "like tensor" in query_string.lower():
        intent = "find_similar_tensors_with_metadata_filter"
        try:
            parts = query_string.lower().split("tensor")
            potential_id_str = parts[1].strip().split(" ")[0]
            params['query_tensor_id_for_lookup'] = potential_id_str
        except IndexError:
            logging.warning("Could not parse tensor ID from similarity query.")
            if context_data and 'query_tensor_data' in context_data:
                 params['query_tensor_data'] = np.array(context_data['query_tensor_data'])
            else:
                return {"error": "Similarity query needs a reference tensor ID or data."}
        
        params['index_id'] = target_index_id
        params['k'] = context_data.get('k', 5) if context_data else 5
        params['metadata_filter'] = context_data.get('metadata_filter', {}) if context_data else {}
        if "tagged with" in query_string.lower():
            try:
                tag_str = query_string.lower().split("tagged with")[1].strip().split(" ")[0]
                params['metadata_filter']['tags'] = tag_str # This was erroring due to '
            except: pass

    elif "find tensors with metadata" in query_string.lower() or "tensors where" in query_string.lower():
        intent = "find_tensors_by_metadata"
        params['metadata_filter'] = context_data.get('metadata_filter', {}) if context_data else {}
        if "user is" in query_string.lower():
            try:
                user_str = query_string.lower().split("user is")[1].strip().split(" ")[0]
                params['metadata_filter']['user_id'] = user_str
            except: pass
        params['retrieve_data'] = context_data.get('retrieve_data', False) if context_data else False
    else:
        logging.warning(f"Could not determine intent from query: '{query_string}'")
        return {"error": "Could not understand agentic query."}

    if not intent:
        return {"error": "Intent not resolved."}

    logging.info(f"Agentic intent: {intent}, params: {params}")
    
    if intent == "find_similar_tensors_with_metadata_filter":
        if not params.get('index_id'):
            return {"error": "No index specified or available for similarity search."}
        if 'query_tensor_id_for_lookup' in params:
            try:
                ref_tensor_data, _ = get_tensor(params['query_tensor_id_for_lookup'])
                params['query_tensor_data'] = ref_tensor_data
            except Exception as e:
                return {"error": f"Could not retrieve reference tensor: {e}"}
        elif 'query_tensor_data' not in params:
             return {"error": "Missing query_tensor_data for similarity search."}
        
        return find_similar_tensors_with_metadata_filter(
            params['index_id'],
            params['query_tensor_data'],
            params['k'],
            params.get('metadata_filter')
        )
    elif intent == "find_tensors_by_metadata":
        return find_tensors_by_metadata(
            params['metadata_filter'],
            params.get('retrieve_data', False)
        )
    
    return {"error": f"Unknown intent '{intent}' after NLU."}

if __name__ == "__main__":
    print("Initializing database...")
    try:
        init_db()
        print("Database initialized.")
    except Exception as e:
        print(f"Could not initialize database. Ensure PostgreSQL is running and configured: {e}")
        print("Please create the database and user manually if they don't exist:")
        print(f"  CREATE DATABASE {DB_NAME};")
        print(f"  CREATE USER {DB_USER} WITH PASSWORD '{DB_PASSWORD}';")
        print(f"  GRANT ALL PRIVILEGES ON DATABASE {DB_NAME} TO {DB_USER};")
        exit(1)

    if os.path.exists(HDF5_DIR):
        for f in os.listdir(HDF5_DIR): os.remove(os.path.join(HDF5_DIR, f))
    if os.path.exists(ANN_INDEX_DIR):
        for f in os.listdir(ANN_INDEX_DIR): os.remove(os.path.join(ANN_INDEX_DIR, f))
    
    print("\n--- Starting Tensorus Example ---")

    print("\n1. Creating Tensors...")
    tensor1_data = np.random.rand(1, 128).astype('float32')
    meta1 = {"name": "image_embedding_001", "source": "model_A", "tags": ["cat", "outdoor"], "user_id": "user1"}
    t1_id = create_tensor(tensor1_data, meta1)
    print(f"  Created tensor t1_id: {t1_id}")

    tensor2_data = np.random.rand(1, 128).astype('float32') + 0.05
    meta2 = {"name": "image_embedding_002", "source": "model_A", "tags": ["cat", "indoor"], "user_id": "user2"}
    t2_id = create_tensor(tensor2_data, meta2)
    print(f"  Created tensor t2_id: {t2_id}")

    tensor3_data = np.random.rand(1, 128).astype('float32') - 0.05
    meta3 = {"name": "doc_embedding_001", "source": "model_B", "tags": ["text", "report"], "user_id": "user1"}
    t3_id = create_tensor(tensor3_data, meta3)
    print(f"  Created tensor t3_id: {t3_id}")

    tensor4_data = np.array([i for i in range(128)]).astype('float32').reshape(1,128) / 128.0
    meta4 = {"name": "audio_embedding_001", "source": "model_C", "tags": ["speech", "meeting"], "user_id": "user3"}
    t4_id = create_tensor(tensor4_data, meta4)
    print(f"  Created tensor t4_id: {t4_id}")
    
    print("\n2. Retrieving a Tensor...")
    ret_tensor_data, ret_meta = get_tensor(t1_id)
    print(f"  Retrieved t1: data shape {ret_tensor_data.shape}, metadata {ret_meta}")
    assert np.array_equal(ret_tensor_data, tensor1_data)
    assert ret_meta == meta1

    print("\n3. Updating a Tensor...")
    updated_meta1 = {"tags": ["cat", "outdoor", "high_res"], "user_id": "user1_updated"}
    updated_tensor1_data = tensor1_data + 0.1
    update_tensor(t1_id, tensor_data=updated_tensor1_data, metadata=updated_meta1)
    ret_updated_tensor_data, ret_updated_meta = get_tensor(t1_id)
    assert np.array_equal(ret_updated_tensor_data, updated_tensor1_data)
    expected_meta1_after_update = meta1.copy()
    expected_meta1_after_update.update(updated_meta1)
    assert ret_updated_meta == expected_meta1_after_update
    print(f"  Updated t1: new metadata {ret_updated_meta}")


    print("\n4. Creating an Index...")
    idx1_id = create_index("image_embeddings_v1", "HNSW", 128, "cosine", {"M": 16, "ef_construction": 100})
    print(f"  Created index idx1_id: {idx1_id}")

    print("\n5. Adding Tensors to Index...")
    add_tensor_to_index(idx1_id, t1_id)
    add_tensor_to_index(idx1_id, t2_id)
    add_tensor_to_index(idx1_id, t3_id)
    print(f"  Added t1, t2, t3 to index {idx1_id}")

    print("\n6. Building the Index...")
    build_index(idx1_id)
    print(f"  Index {idx1_id} built.")

    print("\n7. Querying the Index...")
    query_data_t1 = ret_updated_tensor_data 
    
    query_data_t1_normalized = query_data_t1.copy()
    faiss.normalize_L2(query_data_t1_normalized)
    
    results_t1 = query_index(idx1_id, query_data_t1_normalized, k=3)
    print(f"  Query results for data similar to t1 (expect t1, t2, ...):")
    for res in results_t1:
        print(f"    ID: {res['tensor_id']}, Dist: {res['distance']:.4f}, Meta: {res['metadata'].get('name')}")
    if results_t1:
        assert results_t1[0]['tensor_id'] == str(t1_id)

    print("\n8. Querying with Metadata Filter...")
    results_t1_filtered = query_index(idx1_id, query_data_t1_normalized, k=3, metadata_filter={"tags": "indoor"})
    print(f"  Query results for data similar to t1, filtered by tags='indoor' (expect t2):")
    for res in results_t1_filtered:
        print(f"    ID: {res['tensor_id']}, Dist: {res['distance']:.4f}, Meta: {res['metadata'].get('name')}")
    if results_t1_filtered:
        assert results_t1_filtered[0]['tensor_id'] == str(t2_id)

    print("\n9. Finding Tensors by Metadata...")
    meta_results = find_tensors_by_metadata({"source": "model_A", "user_id": "user2"})
    print(f"  Tensors with source='model_A' AND user_id='user2' (expect t2):")
    for res in meta_results:
        print(f"    ID: {res['tensor_id']}, Meta: {res['metadata']}")
    if meta_results:
        assert meta_results[0]['tensor_id'] == str(t2_id)

    print("\n10. Agentic Query (Simulated NLU)...")
    agent_query_sim = f"Find items similar to tensor {str(t1_id)} tagged with indoor"
    agent_results_sim = agentic_query(agent_query_sim, context_data={'k': 2, 'metadata_filter': {'tags': 'indoor'}}) 
    print(f"  Agentic query: '{agent_query_sim}' with filter {{'tags': 'indoor'}}")
    if isinstance(agent_results_sim, dict) and 'error' in agent_results_sim:
        print(f"  Error: {agent_results_sim['error']}")
    else:
        for res in agent_results_sim: # Expecting list of results
            print(f"    ID: {res['tensor_id']}, Dist: {res['distance']:.4f}, Meta: {res['metadata'].get('name')}")
        if agent_results_sim:
             assert agent_results_sim[0]['tensor_id'] == str(t2_id) # t2 is indoor and similar to t1


    agent_query_meta = "Find tensors where user is user1"
    agent_results_meta = agentic_query(agent_query_meta, context_data={'retrieve_data': False})
    print(f"\n  Agentic query: '{agent_query_meta}'")
    if isinstance(agent_results_meta, dict) and 'error' in agent_results_meta:
        print(f"  Error: {agent_results_meta['error']}")
    else:
        for res in agent_results_meta:
            print(f"    ID: {res['tensor_id']}, Meta: {res['metadata']}")
        assert any(r['tensor_id'] == str(t1_id) for r in agent_results_meta)
        assert any(r['tensor_id'] == str(t3_id) for r in agent_results_meta)


    print("\n11. Deleting a Tensor...")
    delete_tensor(t4_id)
    print(f"  Deleted tensor t4_id: {t4_id}")
    try:
        get_tensor(t4_id) # Should fail
    except KeyError:
        print(f"  Tensor t4_id successfully deleted (get_tensor raised KeyError).")

    print("\n12. Deleting an Index...")
    delete_index(idx1_id)
    print(f"  Deleted index idx1_id: {idx1_id}")
    try:
        get_index_details(idx1_id) # Should fail
    except KeyError:
        print(f"  Index idx1_id successfully deleted (get_index_details raised KeyError).")

    print("\n--- Tensorus Example Completed ---")
