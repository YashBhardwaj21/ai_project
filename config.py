# Configuration settings
import os

# Path configuration
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, 'data')
OUTPUT_DIR = os.path.join(BASE_DIR, 'output')
os.makedirs(OUTPUT_DIR, exist_ok=True)

# File paths
PATIENTS_PATH = os.path.join(DATA_DIR, 'patients.csv')
ENCOUNTERS_PATH = os.path.join(DATA_DIR, 'encounters.csv')
CONDITIONS_PATH = os.path.join(DATA_DIR, 'conditions.csv')
OBSERVATIONS_PATH = os.path.join(DATA_DIR, 'observations.csv')
MEDICATIONS_PATH = os.path.join(DATA_DIR, 'medications.csv')
PROCEDURES_PATH = os.path.join(DATA_DIR, 'procedures.csv')
IMMUNIZATIONS_PATH = os.path.join(DATA_DIR, 'immunizations.csv')
ALLERGIES_PATH = os.path.join(DATA_DIR, 'allergies.csv')

# Output files
EVENT_LOG_PATH = os.path.join(OUTPUT_DIR, 'master_event_log.csv')
SEQUENCES_PATH = os.path.join(OUTPUT_DIR, 'patient_sequences.json')
TRAINING_DATA_PATH = os.path.join(OUTPUT_DIR, 'training_data.pkl')

# Model parameters
MAX_SEQUENCE_LENGTH = 100
VOCAB_SIZE = 10000
EMBEDDING_DIM = 128
BATCH_SIZE = 32
EPOCHS = 10