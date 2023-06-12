

# Custom
FROM_FOLDER = 'mixture_zipf_contexts_nc2_000000' # If a name if provided the files are taken from the folder specified

SRC_NAME ='000000'
TGT_NAME ='000000'

LEXICON_SRC=0
LEXICON_TGT=1

FREQ_SRC=0 # If 0 then uniform freq, else it correspond to the k*10 of the power law (so k = FREQ/10) # Because FREQ cannot be a float :(
FREQ_TGT=0 
PROP_SUPERVISED = 0. # Between 0 and 1
PARA=False # the target is now parallel to the source
FIELD=False # download lexical field data /!\ Freq is k=1.1, no need to specify FREQ_XXX /!\
    
TRANSFORMER = True # Else LSTM
PARTIAL_DICT = False # Train with a partial bilingual dict (half the complete voc)
BILINGUAL_DICT_SUP = False # Train with a supervised bilingual dict
    
# Fix
CODES=1000      # number of BPE codes
N_THREADS=48     # number of threads in data preprocessing
N_EPOCHS=30      # number of fastText epochs

# Other
CUSTOM_FILES = False