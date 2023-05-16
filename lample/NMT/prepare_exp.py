"""
    The goal here is to writte the preprocess of the
    experience in Python to be cleaner.
"""

import os

import config

def get_exp_name(params: dict):
    
    # SRC and TGT names
    SRC_STRING=f"s{params['SRC_NAME']}.LEX{params['LEXICON_SRC']}"
    TGT_STRING=f"t{params['TGT_NAME']}.LEX{params['LEXICON_TGT']}"
    
    # Deals with FROM_FOLDER
    if params['FROM_FOLDER'] != '':
        # First we need to find out which file is which
        params['EXP_NAME'] = f"{params['FROM_FOLDER'].upper()}_{SRC_STRING}.{TGT_STRING}"
        params['SRC_STRING'] = SRC_STRING
        params['TGT_STRING'] = TGT_STRING
        return

    # Add frequency if needed
    if params['FREQ_SRC'] != 0:
        SRC_STRING += f".FREQ_K{params['FREQ_SRC']}" 
    if params['FREQ_TGT'] != 0:
        TGT_STRING += f".FREQ_K{params['FREQ_TGT']}" 

    # Exp name
    EXP_NAME=f"GR_{SRC_STRING}.{TGT_STRING}"

    # Add field and para 
    if params['FIELD']:
        EXP_NAME += '_FIELD'

    if params['PARA']:
        EXP_NAME += '_SUP'
        
    # Update dict
    params['SRC_STRING'] = SRC_STRING
    params['TGT_STRING'] = TGT_STRING
    params['EXP_NAME'] = EXP_NAME
    
    
def download_and_install_tools(paths: dict):
    """
        Download and install tools
    """

    # Download Moses
    if not(os.path.exists(paths['MOSES'])):
        print("Cloning Moses from GitHub repository...")
        os.system(f"cd {paths['TOOLS_PATH']};\
                   git clone https://github.com/moses-smt/mosesdecoder.git")
        print(f"Moses found in: {paths['MOSES']}")
        
    # Download fastBPE
    if not(os.path.exists(paths['FASTBPE_DIR'])):
        print("Cloning fastBPE from GitHub repository...")
        os.system(f"cd {paths['TOOLS_PATH']};\
                   git clone  https://github.com/glample/fastBPE")
        print(f"fastBPE found in: {paths['FASTBPE_DIR']}")

    # Compile fastBPE
    if not(os.path.exists(paths['FASTBPE'])):
        print("Compiling fastBPE...")
        os.system(f"cd {paths['FASTBPE_DIR']};\
                   g++ -std=c++11 -pthread -O3 fastBPE/main.cc -o fast")
        print(f"fastBPE compiled in: {paths['FASTBPE']}")

    # Download fastText
    if not(os.path.exists(paths['FASTTEXT_DIR'])):
        print("Cloning fastText from GitHub repository...")
        os.system(f"cd {paths['TOOLS_PATH']};\
                   git clone https://github.com/facebookresearch/fastText.git")
        print(f"fastText found in: {paths['FASTTEXT_DIR']}")

    # Compile fastText
    if not(os.path.exists(paths['FASTTEXT'])):
        print("Compiling fastText...")
        os.system(f"cd {paths['FASTTEXT_DIR']};\
                   make")
        print(f"fastText compiled in: {paths['FASTTEXT']}")
        
        
def _get_gdown_dict():
    """
        each key is code:
            lex : 0, 1, 2
            source: 0; target: 1
            field: 0 (FIELD=False), 1 (FIELD=True)
            freq: 0 (FREQ=0); 1 (FREQ=11); 2 (FREQ=20)
        
        Remark: For eval we remove the source from the key.
    
        Returns:
            gdown_dict [dict] ex: {'0-0-0-0': {'link': 'https://drive.google.com/uc?id=1tNzIr3JHe0JRXFa87qGnG46V-QrF7dJQ'
                                               'name': 'permuted_samples_source.zip'}}
    """
    
    gdown = {'0-0-0-0': {'link': 'https://drive.google.com/uc?id=1tNzIr3JHe0JRXFa87qGnG46V-QrF7dJQ',
                         'name': 'permuted_samples_source.zip'},
             '0-0-1-1': {'link': 'https://drive.google.com/uc?id=1SnUriwTshjqkXAbBr-XJtqBOksZO7BXA',
                         'name': 'freqk11_permuted_samples_source_fields.zip'},
             '0-0-0-1': {'link': 'https://drive.google.com/uc?id=13CqOmZZ7PtcXfb11Me91Yye2KeEVBwkC',
                         'name': 'freqk11_permuted_samples_source.zip'},
             '0-0-0-2': {'link': 'https://drive.google.com/uc?id=1iM2L1RyFXt5zjQN2Udrr8uuG73JYnYlj',
                         'name': 'freqk2_permuted_samples_source.zip'},
             '0-1-0-0': {'link': 'https://drive.google.com/uc?id=1ak6eXWB054Y3Zg3wQc0n2zjEFdLW4-cm',
                         'name': 'permuted_samples_target.zip'},
             '0-1-1-1': {'link': 'https://drive.google.com/uc?id=1KoAqglDnQOmu8oToJyvWhT6cEnVCqkfA',
                         'name': 'freqk11_permuted_samples_target_fields.zip'},
             '0-1-0-1': {'link': 'https://drive.google.com/uc?id=1NwWyP3_stFjdpN2-yNRQ1YAjCOu9xtX-',
                         'name': 'freqk11_permuted_samples_target.zip'},
             '0-1-0-2': {'link': 'https://drive.google.com/uc?id=1_lvzmqa_hvlBDeiDSZFiew8b6Jd7xYlw',
                         'name': 'freqk2_permuted_samples_target.zip'},
             '1-0-0-0': {'link': 'https://drive.google.com/uc?id=1XM8rPKrEePsJPZYsV_bv7pAF1Iv6pHGz',
                         'name': 'permuted_samples_source_lexicon_1.zip'},
             '1-0-1-1': {'link': 'https://drive.google.com/uc?id=1mdSNy_B1SwWAPxAZmnnhJgMtga8Oi1iK',
                         'name': 'freqk11_permuted_samples_source_fields_lexicon_1.zip'},
             '1-0-0-1': {'link': 'https://drive.google.com/uc?id=17UZ5daDavOKHAiEQx_6KDQYSqE2n7BJ4',
                         'name': 'freqk11_permuted_samples_source_lexicon_1.zip'},
             '1-0-0-2': {'link': 'https://drive.google.com/uc?id=1Rck1Rnzh213yrDe2vAgUzR0Gs9mHDwjG',
                         'name': 'freqk2_permuted_samples_source_lexicon_1.zip'},
             '1-1-0-0': {'link': 'https://drive.google.com/uc?id=16piuveQXb0dACvGMA8PD_aH7X5JSKOyz',
                         'name': 'permuted_samples_target_lexicon_1.zip'},
             '1-1-1-1': {'link': 'https://drive.google.com/uc?id=1X-2Wip7SMeyBxKGUfSiRoBjPZrwL3wpN',
                         'name': 'freqk11_permuted_samples_target_fields_lexicon_1.zip'},
             '1-1-0-1': {'link': 'https://drive.google.com/uc?id=1lByQHzaUwuKxSHU6Yz67DJpAbTF2MCtG',
                         'name': 'freqk11_permuted_samples_target_lexicon_1.zip'},
             '1-1-0-2': {'link': 'https://drive.google.com/uc?id=1QRofOl2QaHuZTjCDmyEnOTo74rUuZrz-',
                         'name': 'freqk2_permuted_samples_target_lexicon_1.zip'},
             '2-0-0-0': {'link': 'https://drive.google.com/uc?id=1gF6tv5AradQn1AW2HQzQBZf7VZef6aaH',
                         'name': 'permuted_samples_source_lexicon_2.zip'},
             '2-1-0-0': {'link': 'https://drive.google.com/uc?id=1JCxht0OR_ZVCU70nxQLYfDBLKNdRm6Ab',
                         'name': 'permuted_samples_target_lexicon_2.zip'},
             'testperanto': {'link': 'https://drive.google.com/uc?id=1n-tmFz4rOgXYHLPkwlf29qzO9jyHNNOI',
                             'name': 'testperanto.zip'}
             }
    
    gdown_valid = {'0-0-0': {'link': 'https://drive.google.com/uc?id=10_j0MK8jiOe8C0JvAuPz-Dw5Zm_Lqkn3',
                             'name': 'permuted_samples_valid.zip'},
                   '0-1-1': {'link': 'https://drive.google.com/uc?id=18DzgZqvvIs5rYJeqpgbUfO_3B42T7oAb',
                             'name': 'freqk11_permuted_samples_valid_fields.zip'},
                   '0-0-1': {'link': 'https://drive.google.com/uc?id=1gJNqMuVznjHjcObXiqwxk-7t-6RMgjwn',
                             'name': 'freqk11_permuted_samples_valid.zip'},
                   '0-0-2': {'link': 'https://drive.google.com/uc?id=12YrU6YTMnjdpzxfu9esdwLUb51i5pMAG',
                             'name': 'freqk2_permuted_samples_valid.zip'},
                   '1-0-0': {'link': 'https://drive.google.com/uc?id=17bcTIJA8lNyZDhdBKhAef2FCwstgCB12',
                             'name': 'permuted_samples_valid_lexicon_1.zip'},
                   '1-1-1': {'link': 'https://drive.google.com/uc?id=1r-mT7oneUj9tEGHL6lrOe1DZU5JovUL7',
                             'name': 'freqk11_permuted_samples_valid_fields_lexicon_1.zip'},
                   '1-0-1': {'link': 'https://drive.google.com/uc?id=1rtBYeyBN86tu1U3wak6KZMaoRJ0_Sgc-',
                             'name': 'freqk11_permuted_samples_valid_lexicon_1.zip'},
                   '1-0-2': {'link': 'https://drive.google.com/uc?id=1TOYIkrgJs-bsORR-vk2OD3Wg2ajHSOPi',
                             'name': 'freqk2_permuted_samples_valid_lexicon_1.zip'},
                   '2-0-0': {'link': 'https://drive.google.com/uc?id=15jG-1GOBRW1upzMrW6vivoC1ZT32ZDVm',
                             'name': 'permuted_samples_valid_lexicon_2.zip'},
                   }
    
    gdown_test =  {'0-0-0': {'link': 'https://drive.google.com/uc?id=1zd_fL7RIfCp8YE9zz8tMIbUuOiBROd8P',
                             'name': 'permuted_samples_test.zip'},
                   '0-1-1': {'link': 'https://drive.google.com/uc?id=1WAUw1HM7Va_HT15OCDsrDFT6sk-IEd4Z',
                             'name': 'freqk11_permuted_samples_test_fields.zip'},
                   '0-0-1': {'link': 'https://drive.google.com/uc?id=1eIoZJ5tb19H0prDcaVwx1nI936onRnoQ',
                             'name': 'freqk11_permuted_samples_test.zip'},
                   '0-0-2': {'link': 'https://drive.google.com/uc?id=1msZmwX6jEXyMz7ukOI2QkcxQwOhefXNk',
                             'name': 'freqk2_permuted_samples_test.zip'},
                   '1-0-0': {'link': 'https://drive.google.com/uc?id=1g0viMLTUz5XNkuQHd-l1NvmiLjgVLvGq',
                             'name': 'permuted_samples_test_lexicon_1.zip'},
                   '1-1-1': {'link': 'https://drive.google.com/uc?id=1PYoT2nhSApJ7HXXxe1wlGEe2JFqdYY1C',
                             'name': 'freqk11_permuted_samples_test_fields_lexicon_1.zip'},
                   '1-0-1': {'link': 'https://drive.google.com/uc?id=1LeJEqT0LK9WAVgVzusbkf9vIR1ZK2xAl',
                             'name': 'freqk11_permuted_samples_test_lexicon_1.zip'},
                   '1-0-2': {'link': 'https://drive.google.com/uc?id=1aaai_B0KBuY6QCZFKPrgB1-e9v84-0CU',
                             'name': 'freqk2_permuted_samples_test_lexicon_1.zip'},
                   '2-0-0': {'link': 'https://drive.google.com/uc?id=1KLhkOFYGf_gFmgQ3k1ksoAdAOFmMiqWO',
                             'name': 'permuted_samples_test_lexicon_2.zip'},
                   }
    
    
    return gdown, gdown_valid, gdown_test

def _get_keys(params: dict):
    """
    
    """
    if params['FIELD']:
        field = 1
    else:
        field = 0
    if params['FREQ_SRC'] == 0:
        freq = 0
    elif params['FREQ_SRC'] == 11:
        freq = 1
    elif params['FREQ_SRC'] == 20:
        freq = 2
        
    key_source = f"{params['LEXICON_SRC']}-0-{field}-{freq}"
    key_source_eval = f"{params['LEXICON_SRC']}-{field}-{freq}"
    
    if params['FREQ_TGT'] == 0:
        freq = 0
    elif params['FREQ_TGT'] == 11:
        freq = 1
    elif params['FREQ_TGT'] == 20:
        freq = 2
        
    if params['PARA']:
        split = 0
    else:
        split = 1
    
    key_target = f"{params['LEXICON_TGT']}-{split}-{field}-{freq}"
    key_target_eval = f"{params['LEXICON_TGT']}-{field}-{freq}"
    
    return key_source, key_target, key_source_eval, key_source_eval
               
def download_raws(paths: dict,
                  key_source: str = '',
                  key_target: str = '',
                  key_source_eval: str = '',
                  key_target_eval: str = ''):
    """
    
    
    """
    
    if key_target == '': # Dealing with FROM_FOLDER
        assert key_source_eval == ''
        assert key_target_eval == ''
        
        if not(os.path.exists(paths['RAWS_SRC'])):
            try:     
                # Instal gdown
                os.system("pip install gdown")
                # Create dir
                os.makedirs(paths['RAWS_SRC'],
                            exist_ok=True)
            
                gdown, _, _ = _get_gdown_dict()
            
                # In this setup key_source is the name of the folder
                link = gdown[key_source]['link']
                name =  gdown[key_source]['name']
                
                os.system(f"cd {paths['RAWS_SRC']};\
                            gdown {link};\
                            unzip -j {name}")
            except:
                raise Exception("The folder from which you want to train doesn't exist.")
        
        return
    
    
    # SOURCE
    if not(os.path.exists(paths['RAWS_SRC'])):
        # Instal gdown
        os.system("pip install gdown")
        # Create dir
        os.makedirs(paths['RAWS_SRC'],
                    exist_ok=True)
            
        #Download what is needed
        gdown, _, _ = _get_gdown_dict()
        
        link = gdown[key_source]['link']
        name =  gdown[key_source]['name']
        
        os.system(f"cd {paths['RAWS_SRC']};\
                    gdown {link};\
                    unzip -j {name}")
        
    if not(os.path.exists(paths['RAWS_VALID_SRC'])):
        
        assert not(os.path.exists(paths['RAWS_TEST_SRC']))
        
        # Instal gdown
        os.system("pip install gdown")
        # Create dir
        os.makedirs(paths['RAWS_VALID_SRC'],
                    exist_ok=True)
        os.makedirs(paths['RAWS_TEST_SRC'],
                    exist_ok=True)
            
        #Download what is needed
        _, gdown_valid, gdown_test = _get_gdown_dict()
        
        link_valid = gdown_valid[key_source_eval]['link']
        name_valid =  gdown_valid[key_source_eval]['name']
        link_test = gdown_test[key_source_eval]['link']
        name_test =  gdown_test[key_source_eval]['name']
        
        os.system(f"cd {paths['RAWS_VALID_SRC']};\
                    gdown {link_valid};\
                    unzip -j {name_valid}")
        os.system(f"cd {paths['RAWS_TEST_SRC']};\
                    gdown {link_test};\
                    unzip -j {name_test}")
    
    # TGT    
    if not(os.path.exists(paths['RAWS_TGT'])):
        # Instal gdown
        os.system("pip install gdown")
        # Create dir
        os.makedirs(paths['RAWS_TGT'],
                    exist_ok=True)
            
        #Download what is needed
        gdown, _, _ = _get_gdown_dict()
        
        link = gdown[key_target]['link']
        name =  gdown[key_target]['name']
        
        os.system(f"cd {paths['RAWS_TGT']};\
                    gdown {link};\
                    unzip -j {name}")
        
    if not(os.path.exists(paths['RAWS_VALID_TGT'])):
        
        assert not(os.path.exists(paths['RAWS_TEST_TGT']))
        
        # Instal gdown
        os.system("pip install gdown")
        # Create dir
        os.makedirs(paths['RAWS_VALID_TGT'],
                    exist_ok=True)
        os.makedirs(paths['RAWS_TEST_TGT'],
                    exist_ok=True)
            
        #Download what is needed
        _, gdown_valid, gdown_test = _get_gdown_dict()
        
        link_valid = gdown_valid[key_target_eval]['link']
        name_valid =  gdown_valid[key_target_eval]['name']
        link_test = gdown_test[key_target_eval]['link']
        name_test =  gdown_test[key_target_eval]['name']
        
        os.system(f"cd {paths['RAWS_VALID_TGT']};\
                    gdown {link_valid};\
                    unzip -j {name_valid}")
        os.system(f"cd {paths['RAWS_TEST_TGT']};\
                    gdown {link_test};\
                    unzip -j {name_test}")
        
def preprocess_raws(paths: dict,
                    params: dict):
    """
    
    """
    
    # Create EXP folder
    
    paths['EXP'] = os.path.join(paths['EXPS_PATH'],
                                params['EXP_NAME'])
    if os.path.exists(paths['EXP']):
        print("Experiment already exists.")
        
    os.makedirs(paths['EXP'],
                exist_ok=True)
    
    # Usefull paths
    if params['FROM_FOLDER'] == '':
        paths['SRC_RAW'] = os.path.join(paths['RAWS_SRC'],
                                        f"sample_{params['SRC_NAME']}.txt")
        paths['TGT_RAW'] = os.path.join(paths['RAWS_TGT'],
                                        f"sample_{params['TGT_NAME']}.txt")
        
        paths['SRC_VALID_RAW'] = os.path.join(paths['RAWS_VALID_SRC'],
                                            f"sample_{params['SRC_NAME']}.txt")
        paths['TGT_VALID_RAW'] = os.path.join(paths['RAWS_VALID_TGT'],
                                            f"sample_{params['TGT_NAME']}.txt")
        paths['SRC_TEST_RAW'] = os.path.join(paths['RAWS_TEST_SRC'],
                                            f"sample_{params['SRC_NAME']}.txt")
        paths['TGT_TEST_RAW'] = os.path.join(paths['RAWS_TEST_TGT'],
                                            f"sample_{params['TGT_NAME']}.txt")
    else:
        # We need to find which file is which
        files = os.listdir(path = paths['RAWS_SRC']) # each 'RAWS_XXX' are the same aka the FROM_FOLDER
        
        for file in files:
            if 'src' in file and f"lex{params['LEXICON_SRC']}" in file:
                paths['SRC_RAW'] = os.path.join(paths['RAWS_SRC'],
                                                file)
            if 'tgt' in file and f"lex{params['LEXICON_TGT']}" in file:
                paths['TGT_RAW'] = os.path.join(paths['RAWS_SRC'],
                                                file)
            if 'valid' in file and f"lex{params['LEXICON_SRC']}" in file:
                paths['SRC_VALID_RAW'] = os.path.join(paths['RAWS_SRC'],
                                                      file)
            if 'valid' in file and f"lex{params['LEXICON_TGT']}" in file:
                paths['TGT_VALID_RAW'] = os.path.join(paths['RAWS_SRC'],
                                                      file)

            if 'test' in file and f"lex{params['LEXICON_SRC']}" in file:
                paths['SRC_TEST_RAW'] = os.path.join(paths['RAWS_SRC'],
                                                      file)
            if 'test' in file and f"lex{params['LEXICON_TGT']}" in file:
                paths['TGT_TEST_RAW'] = os.path.join(paths['RAWS_SRC'],
                                                      file)
    
    # Print summary
    print(f"SRC_RAW {paths['SRC_RAW']}")
    print(f"TGT_RAW {paths['TGT_RAW']}")
    print(f"SRC_VALID_RAW {paths['SRC_VALID_RAW']}")
    print(f"TGT_VALID_RAW {paths['TGT_VALID_RAW']}")
    print(f"SRC_TEST_RAW {paths['SRC_TEST_RAW']}")
    print(f"TGT_TEST_RAW {paths['TGT_TEST_RAW']}")
    
    paths['SRC_TOK']=os.path.join(paths['EXP'],
                                  f"{params['SRC_STRING']}.tok")
    paths['TGT_TOK']=os.path.join(paths['EXP'],
                                  f"{params['TGT_STRING']}.tok")
    
    paths['SRC_VALID_TOK']=os.path.join(paths['EXP'],
                                        f"{params['SRC_STRING']}_VALID.tok")
    paths['TGT_VALID_TOK']=os.path.join(paths['EXP'],
                                        f"{params['TGT_STRING']}_VALID.tok")
    paths['SRC_TEST_TOK']=os.path.join(paths['EXP'],
                                        f"{params['SRC_STRING']}_TEST.tok")
    paths['TGT_TEST_TOK']=os.path.join(paths['EXP'],
                                        f"{params['TGT_STRING']}_TEST.tok")
    
    paths['BPE_CODES']=os.path.join(paths['EXP'],
                                    f"bpe_codes{params['CODES']}")
    paths['CONCAT_BPE']=os.path.join(paths['EXP'],
                                     f"all.{params['CODES']}")

    paths['SRC_VOCAB']=os.path.join(paths['EXP'],
                                    f"vocab.SRC.{params['CODES']}")
    paths['TGT_VOCAB']=os.path.join(paths['EXP'],
                                    f"vocab.TGT.{params['CODES']}")
    paths['FULL_VOCAB']=os.path.join(paths['EXP'],
                                     f"vocab.{params['CODES']}")
        
    # tokenize data
    if not(os.path.exists(paths['SRC_TOK'])) or not(os.path.exists(paths['TGT_TOK'])):
        print("Tokenize monolingual data...")
        os.system(
            f"cat {paths['SRC_RAW']} | {paths['NORM_PUNC']} -l en | {paths['TOKENIZER']} -l en -no-escape -threads {params['N_THREADS']} > {paths['SRC_TOK']};\
              cat {paths['TGT_RAW']} | {paths['NORM_PUNC']} -l en | {paths['TOKENIZER']} -l en -no-escape -threads {params['N_THREADS']} > {paths['TGT_TOK']};"
        )
    
    print(f"{params['SRC_NAME']} monolingual data tokenized in: {paths['SRC_TOK']}")
    print(f"{params['TGT_NAME']} monolingual data tokenized in: {paths['TGT_TOK']}")

    # learn BPE codes
    if not(os.path.exists(paths['BPE_CODES'])):
        print("Learning BPE codes...")
        os.system(f"{paths['FASTBPE']} learnbpe {params['CODES']} {paths['SRC_TOK']} {paths['TGT_TOK']} > {paths['BPE_CODES']}")
    print(f"BPE learned in {paths['BPE_CODES']}")

    # apply BPE codes
    if not(os.path.exists(f"{paths['SRC_TOK']}.{params['CODES']}")) or \
       not(os.path.exists(f"{paths['TGT_TOK']}.{params['CODES']}")):
        print(f"Applying BPE codes...")
        os.system(f"{paths['FASTBPE']} applybpe {paths['SRC_TOK']}.{params['CODES']} {paths['SRC_TOK']} {paths['BPE_CODES']}")
        os.system(f"{paths['FASTBPE']} applybpe {paths['TGT_TOK']}.{params['CODES']} {paths['TGT_TOK']} {paths['BPE_CODES']}")
    
    print(f"BPE codes applied to {params['SRC_NAME']} in: {paths['SRC_TOK']}.{params['CODES']}")
    print(f"BPE codes applied to {params['TGT_NAME']} in: {paths['TGT_TOK']}.{params['CODES']}")

    # extract vocabulary
    if not(os.path.exists(paths['SRC_VOCAB'])) or \
       not(os.path.exists(paths['TGT_VOCAB'])) or\
       not(os.path.exists(paths['FULL_VOCAB'])):
        print(f"Extracting vocabulary...")
        os.system(f"{paths['FASTBPE']} getvocab {paths['SRC_TOK']}.{params['CODES']} > {paths['SRC_VOCAB']}")
        os.system(f"{paths['FASTBPE']} getvocab {paths['TGT_TOK']}.{params['CODES']} > {paths['TGT_VOCAB']}")
        os.system(f"{paths['FASTBPE']} getvocab {paths['SRC_TOK']}.{params['CODES']} {paths['TGT_TOK']}.{params['CODES']} > {paths['FULL_VOCAB']}")
    
    print(f"{params['SRC_NAME']} vocab in: {paths['SRC_VOCAB']}")
    print(f"{params['TGT_NAME']} vocab in: {paths['TGT_VOCAB']}")
    print(f"Full vocab in: {paths['FULL_VOCAB']}")

    # binarize data
    if not(os.path.exists(f"{paths['SRC_TOK']}.{params['CODES']}.pth")) or \
       not(os.path.exists(f"{paths['TGT_TOK']}.{params['CODES']}.pth")):
        print(f"Binarizing data...")
        os.system(f"{paths['UMT_PATH']}/preprocess.py {paths['FULL_VOCAB']} {paths['SRC_TOK']}.{params['CODES']}")
        os.system(f"{paths['UMT_PATH']}/preprocess.py {paths['FULL_VOCAB']} {paths['TGT_TOK']}.{params['CODES']}")
    
    print(f"{params['SRC_NAME']} binarized data in: {paths['SRC_TOK']}.{params['CODES']}.pth")
    print(f"{params['TGT_NAME']} binarized data in: {paths['TGT_TOK']}.{params['CODES']}.pth") 
    
    
    # eval data
    # tokenize data
    if not(os.path.exists(paths['SRC_VALID_TOK'])) or not(os.path.exists(paths['TGT_VALID_TOK'])):
        print("Tokenize valid data...")
        os.system(
            f"cat {paths['SRC_VALID_RAW']} | {paths['NORM_PUNC']} -l en | {paths['TOKENIZER']} -l en -no-escape -threads {params['N_THREADS']} > {paths['SRC_VALID_TOK']};\
              cat {paths['TGT_VALID_RAW']} | {paths['NORM_PUNC']} -l en | {paths['TOKENIZER']} -l en -no-escape -threads {params['N_THREADS']} > {paths['TGT_VALID_TOK']};"
        )
        
    if not(os.path.exists(paths['SRC_TEST_TOK'])) or not(os.path.exists(paths['TGT_TEST_TOK'])):
        print("Tokenize test data...")
        os.system(
            f"cat {paths['SRC_TEST_RAW']} | {paths['NORM_PUNC']} -l en | {paths['TOKENIZER']} -l en -no-escape -threads {params['N_THREADS']} > {paths['SRC_TEST_TOK']};\
              cat {paths['TGT_TEST_RAW']} | {paths['NORM_PUNC']} -l en | {paths['TOKENIZER']} -l en -no-escape -threads {params['N_THREADS']} > {paths['TGT_TEST_TOK']};"
        )

    print("Applying BPE to valid and test files...")
    os.system(f"{paths['FASTBPE']} applybpe {paths['SRC_VALID_TOK']}.{params['CODES']} {paths['SRC_VALID_TOK']} {paths['BPE_CODES']} {paths['SRC_VOCAB']}")
    os.system(f"{paths['FASTBPE']} applybpe {paths['TGT_VALID_TOK']}.{params['CODES']} {paths['TGT_VALID_TOK']} {paths['BPE_CODES']} {paths['TGT_VOCAB']}")
    os.system(f"{paths['FASTBPE']} applybpe {paths['SRC_TEST_TOK']}.{params['CODES']} {paths['SRC_TEST_TOK']} {paths['BPE_CODES']} {paths['SRC_VOCAB']}")
    os.system(f"{paths['FASTBPE']} applybpe {paths['TGT_TEST_TOK']}.{params['CODES']} {paths['TGT_TEST_TOK']} {paths['BPE_CODES']} {paths['TGT_VOCAB']}")

    print("Binarizing data...")
    # delete previous versions
    os.system(f"rm -f {paths['SRC_VALID_TOK']}.{params['CODES']}.pth")
    os.system(f"rm -f {paths['TGT_VALID_TOK']}.{params['CODES']}.pth")
    os.system(f"rm -f {paths['SRC_TEST_TOK']}.{params['CODES']}.pth")
    os.system(f"rm -f {paths['TGT_TEST_TOK']}.{params['CODES']}.pth")
    # create .pth
    os.system(f"{paths['UMT_PATH']}/preprocess.py {paths['FULL_VOCAB']} {paths['SRC_VALID_TOK']}.{params['CODES']}")
    os.system(f"{paths['UMT_PATH']}/preprocess.py {paths['FULL_VOCAB']} {paths['TGT_VALID_TOK']}.{params['CODES']}")
    os.system(f"{paths['UMT_PATH']}/preprocess.py {paths['FULL_VOCAB']} {paths['SRC_TEST_TOK']}.{params['CODES']}")
    os.system(f"{paths['UMT_PATH']}/preprocess.py {paths['FULL_VOCAB']} {paths['TGT_TEST_TOK']}.{params['CODES']}")
    
    
def preprocess_additional_files(paths: dict,
                                params: dict):
    """
    
    """
    
    # Download and unzip
    
    if not(os.path.exists(paths['CUSTOMIZED'])):
        
        link = "https://drive.google.com/uc?id=1VcVvegAFp3ZT-zRRnRjFrcZF7qQop_sB"
        name = "additional_files.zip"
        
        os.system(f"cd {paths['GRAMMARS_PATH']};\
                    gdown {link};\
                    unzip -j {name}")
        exit(0)
    
    # tokenize data
    print("Tokenizing customized data...")
    os.system(f"cat {paths['CUSTOMIZED_RAW']} | {paths['NORM_PUNC']} -l en | {paths['TOKENIZER']} -l en -no-escape -threads {params['N_THREADS']} > {paths['CUSTOMIZED_TOK']}")
    if 'LEXICON0_RAW' in paths.keys():
        os.system(f"cat {paths['LEXICON0_RAW']} | {paths['NORM_PUNC']} -l en | {paths['TOKENIZER']} -l en -no-escape -threads {params['N_THREADS']} > {paths['LEXICON0_TOK']}")
        os.system(f"cat {paths['LEXICON1_RAW']} | {paths['NORM_PUNC']} -l en | {paths['TOKENIZER']} -l en -no-escape -threads {params['N_THREADS']} > {paths['LEXICON1_TOK']}")

    
    # apply fastbpe
    print("Applying FastBPE...")
    os.system(f"{paths['FASTBPE']} applybpe {paths['CUSTOMIZED_TOK']}.{params['CODES']} {paths['CUSTOMIZED_TOK']} {paths['BPE_CODES']} {paths['SRC_VOCAB']}")
    if 'LEXICON0_RAW' in paths.keys():
        os.system(f"{paths['FASTBPE']} applybpe {paths['LEXICON0_TOK']}.{params['CODES']} {paths['LEXICON0_TOK']} {paths['BPE_CODES']} {paths['SRC_VOCAB']}")
        os.system(f"{paths['FASTBPE']} applybpe {paths['LEXICON1_TOK']}.{params['CODES']} {paths['LEXICON1_TOK']} {paths['BPE_CODES']} {paths['TGT_VOCAB']}")
    
    # Binarizing data...
    print("Binarizing customized data...")
    os.system(f"rm -f {paths['CUSTOMIZED_TOK']}.{params['CODES']}.pth")
    os.system(f"{paths['UMT_PATH']}/preprocess.py {paths['FULL_VOCAB']} {paths['CUSTOMIZED_TOK']}.{params['CODES']}")
    if 'LEXICON0_RAW' in paths.keys():
        os.system(f"rm -f {paths['LEXICON0_TOK']}.{params['CODES']}.pth")
        os.system(f"{paths['UMT_PATH']}/preprocess.py {paths['FULL_VOCAB']} {paths['LEXICON0_TOK']}.{params['CODES']}")    
        os.system(f"rm -f {paths['LEXICON1_TOK']}.{params['CODES']}.pth")
        os.system(f"{paths['UMT_PATH']}/preprocess.py {paths['FULL_VOCAB']} {paths['LEXICON1_TOK']}.{params['CODES']}")
    

if __name__ == '__main__':
    
    ### PARAMS ###
    
    params = {}
    
    # Custom
    params['FROM_FOLDER'] = config.FROM_FOLDER
    
    params['SRC_NAME']=config.SRC_NAME
    params['TGT_NAME']=config.TGT_NAME
    params['LEXICON_SRC']=config.LEXICON_SRC
    params['LEXICON_TGT']=config.LEXICON_TGT
    params['FREQ_SRC']=config.FREQ_SRC # If 0 then uniform freq, else it correspond to the k*10 of the power law (so k = FREQ/10) # Because FREQ cannot be a float :(
    params['FREQ_TGT']=config.FREQ_TGT 
    params['PARA']=config.PARA # the target is now parallel to the source
    params['FIELD']=config.FIELD # download lexical field data /!\ Freq is k=1.1, no need to specify FREQ_XXX /!\
    
    assert params['LEXICON_SRC'] == 0
    
    # Fix
    params['CODES']=config.CODES      # number of BPE codes
    params['N_THREADS']=config.N_THREADS     # number of threads in data preprocessing
    params['N_EPOCHS']=config.N_EPOCHS      # number of fastText epochs
    
    get_exp_name(params)
    
    ### PATHS ###
    
    paths = {}
    
    # Main Paths
    paths['UMT_PATH']=os.getcwd()
    paths['TOOLS_PATH']=os.path.join(os.getcwd(), 
                                     "tools")
    paths['DATA_PATH']=os.path.join(os.getcwd(), 
                                    "data")
    paths['GRAMMARS_PATH']=os.path.join(paths['DATA_PATH'], 
                                        "artificial_grammars")
    paths['EXPS_PATH']=os.path.join(paths['GRAMMARS_PATH'], 
                                          "EXPS")
    paths['RAWS_PATH']=os.path.join(paths['GRAMMARS_PATH'], 
                                          "RAWS")
    
    # moses
    paths['MOSES']=os.path.join(paths['TOOLS_PATH'], 
                                "mosesdecoder")
    paths['TOKENIZER']=os.path.join(paths['MOSES'], 
                                    "scripts", 
                                    "tokenizer", 
                                    "tokenizer.perl")
    paths['NORM_PUNC']=os.path.join(paths['MOSES'], 
                                    "scripts", 
                                    "tokenizer", 
                                    "normalize-punctuation.perl")
    paths['REM_NON_PRINT_CHAR']=os.path.join(paths['MOSES'],
                                             "scripts",
                                             "tokenizer",
                                             "remove-non-printing-char.perl")

    # fastBPE
    paths['FASTBPE_DIR']=os.path.join(paths['TOOLS_PATH'],
                                      "fastBPE")
    paths['FASTBPE']=os.path.join(paths['FASTBPE_DIR'],
                                  "fast")

    # fastText
    paths['FASTTEXT_DIR']=os.path.join(paths['TOOLS_PATH'], 
                                       "fastText")
    paths['FASTTEXT']=os.path.join(paths['FASTTEXT_DIR'], 
                                   "fasttext")
    
    
    ### CREATE DIR ###
    
    os.makedirs(paths['TOOLS_PATH'], 
                exist_ok=True)
    os.makedirs(paths['GRAMMARS_PATH'], 
                exist_ok=True)
    
    
    ### INSTALL TOOLS ###
    
    download_and_install_tools(paths)
    
    
    ### DOWNLOAD DATA ###
    
    # Get keys that correspond to this experiment
    if params['FROM_FOLDER'] == '':
        print("No FROM_FOLDER")
        key_source, key_target, key_source_eval, key_target_eval = _get_keys(params)
        
        # Create associated paths
        paths['RAWS_SRC'] = os.path.join(paths['RAWS_PATH'],
                                        key_source)
        paths['RAWS_TGT'] = os.path.join(paths['RAWS_PATH'],
                                        key_target)
        paths['RAWS_VALID_SRC'] = os.path.join(paths['RAWS_PATH'],
                                            "VALID",
                                            key_source_eval)
        paths['RAWS_VALID_TGT'] = os.path.join(paths['RAWS_PATH'],
                                            "VALID",
                                            key_target_eval)
        paths['RAWS_TEST_SRC'] = os.path.join(paths['RAWS_PATH'],
                                            "TEST",
                                            key_source_eval)
        paths['RAWS_TEST_TGT'] = os.path.join(paths['RAWS_PATH'],
                                            "TEST",
                                            key_target_eval)
        
        download_raws(paths,
                      key_source,
                      key_target,
                      key_source_eval,
                      key_target_eval)
    else:
        paths['RAWS_SRC'] = os.path.join(paths['RAWS_PATH'],
                                         params['FROM_FOLDER'].upper())
        paths['RAWS_TGT'] = paths['RAWS_SRC']
        paths['RAWS_VALID_SRC'] = paths['RAWS_SRC']
        paths['RAWS_VALID_TGT'] = paths['RAWS_SRC']
        paths['RAWS_TEST_SRC'] = paths['RAWS_SRC']
        paths['RAWS_TEST_TGT'] = paths['RAWS_SRC']
        
        download_raws(paths,
                      params['FROM_FOLDER'])
    
    
    
    ### PROCESS DATA ###
    
    preprocess_raws(paths,
                    params)
    
    #
    # Summary
    #
    print("")
    print("===== Data summary")
    print("Monolingual training data:")
    print(f"    {params['SRC_NAME']}: {paths['SRC_TOK']}.{params['CODES']}.pth")
    print(f"    {params['TGT_NAME']}: {paths['TGT_TOK']}.{params['CODES']}.pth")
    print("Parallel validation data:")
    print(f"    {params['SRC_NAME']}: {paths['SRC_VALID_TOK']}.{params['CODES']}.pth")
    print(f"    {params['TGT_NAME']}: {paths['SRC_VALID_TOK']}.{params['CODES']}.pth")
    print("Parallel test data:")
    print(f"    {params['SRC_NAME']}: {paths['SRC_TEST_TOK']}.{params['CODES']}.pth")
    print(f"    {params['TGT_NAME']}: {paths['SRC_TEST_TOK']}.{params['CODES']}.pth")
    print("")
    
    #
    # Train fastText on concatenated embeddings
    #
    
    if not(os.path.exists(paths['CONCAT_BPE'])):
        print("Concatenating source and target monolingual data...")
        os.system(f"cat {paths['SRC_TOK']}.{params['CODES']} {paths['TGT_TOK']}.{params['CODES']} | shuf > {paths['CONCAT_BPE']}")
    print(f"Concatenated data in: {paths['CONCAT_BPE']}")

    if not(os.path.exists(f"{paths['CONCAT_BPE']}.vec")):
        print("Training fastText on $CONCAT_BPE...")
        os.system(f"{paths['FASTTEXT']} skipgram -epoch {params['N_EPOCHS']} -minCount 0 -dim 512 -thread {params['N_THREADS']} -ws 5 -neg 10 -input {paths['CONCAT_BPE']} -output {paths['CONCAT_BPE']}")
    print(f"Cross-lingual embeddings in: {paths['CONCAT_BPE']}.vec")

    # Deletin bin file because it is too big!
    print(f"Deleting {paths['CONCAT_BPE']}.bin")
    os.system(f"rm -r -f {paths['CONCAT_BPE']}.bin")
    print("Done!")
    
    
    ### RUN EXP ###
    
    params['PROP_SUPERVISED'] = config.PROP_SUPERVISED
    params['BILINGUAL_DICT_SUP'] = config.BILINGUAL_DICT_SUP
    params['CUSTOM_FILES'] = config.CUSTOM_FILES
    
    params['TRANSFORMER'] = config.TRANSFORMER
    
    
    # Customized Files and Bilingual Dict
    if params['BILINGUAL_DICT_SUP'] or params['CUSTOM_FILES']:
        paths['CUSTOMIZED'] = os.path.join(paths['GRAMMARS_PATH'],
                                        "CUSTOMIZED")
        paths['BILINGUAL_DICT'] = os.path.join(paths['GRAMMARS_PATH'],
                                            "BILINGUAL_DICT")
        
        paths['CUSTOMIZED_RAW'] = os.path.join(paths['CUSTOMIZED'],
                                            "lexicon_0_words_only.txt")
        paths['CUSTOMIZED_TOK'] = os.path.join(paths['CUSTOMIZED'],
                                            "lexicon_0_words_only.tok")

        if params['BILINGUAL_DICT_SUP']:
            
            assert params['LEXICON_SRC'] == 0
            assert params['LEXICON_TGT'] == 1
            
            if params['PARTIAL_DICT']:
                print("Partial bilingual dict!")
                params['EXP_NAME']+='_PARTIAL'
                paths['LEXICON0_RAW']=os.path.join(paths['BILINGUAL_DICT'],
                                                "lexicon_0_words_only_partial.txt")
                paths['LEXICON0_TOK']=os.path.join(paths['BILINGUAL_DICT'],
                                                f"{params['SRC_STRING']}.tok")
                paths['LEXICON1_RAW']=os.path.join(paths['BILINGUAL_DICT'],
                                                "lexicon_1_words_only_partial.txt")
                paths['LEXICON1_TOK']=os.path.join(paths['BILINGUAL_DICT'],
                                                f"{params['TGT_STRING']}.tok")
            else:
                print("Full bilingual dict!")
                paths['LEXICON0_RAW']=os.path.join(paths['BILINGUAL_DICT'],
                                                "lexicon_0_words_only.txt")
                paths['LEXICON0_TOK']=os.path.join(paths['BILINGUAL_DICT'],
                                                f"{params['SRC_STRING']}.tok")
                paths['LEXICON1_RAW']=os.path.join(paths['BILINGUAL_DICT'],
                                                "lexicon_1_words_only.txt")
                paths['LEXICON1_TOK']=os.path.join(paths['BILINGUAL_DICT'],
                                                f"{params['TGT_STRING']}.tok")
                
        preprocess_additional_files(paths, params)
        
        custom_file_arg = f"--customized_data {params['SRC_STRING']}:{paths['CUSTOMIZED_TOK']}.{params['CODES']}.pth\;{params['TGT_STRING']}:"
    else:
        custom_file_arg = ""
    
    
    # Run main.py
    
    if params['PROP_SUPERVISED'] == 1.:
        print('Supervised Training!')
        os.system(f"python main.py\
                    --exp_name {params['EXP_NAME']}\
                    --batch_size 16\
                    --wandb\
                    --max_epoch 50\
                    --transformer {params['TRANSFORMER']}\
                    --n_enc_layers 4 \
                    --n_dec_layers 4 \
                    --share_enc 3 \
                    --share_dec 3 \
                    --share_lang_emb True \
                    --share_output_emb True \
                    --langs {params['SRC_STRING']},{params['TGT_STRING']} \
                    --n_para -1 \
                    --para_dataset {params['SRC_STRING']}-{params['TGT_STRING']}:{paths['EXP']}/XX.tok.{params['CODES']}.pth,\
                                                                                 {paths['EXP']}/XX_VALID.tok.{params['CODES']}.pth,\
                                                                                 {paths['EXP']}/XX_TEST.tok.{params['CODES']}.pth \
                    --para_directions {params['SRC_STRING']}-{params['TGT_STRING']},{params['TGT_STRING']}-{params['SRC_STRING']} \
                    --pretrained_emb {paths['CONCAT_BPE']}.vec \
                    --pretrained_out True \
                    --lambda_xe_para '0:1,100000:0.1,300000:0' \
                    --otf_num_processes 30 \
                    --otf_sync_params_every 1000 \
                    --enc_optimizer adam,lr=0.0001 \
                    --epoch_size 100000 \
                    --stopping_criterion 'bleu_{params['SRC_STRING']}_{params['TGT_STRING']}_valid,20' \
                    --customized_data {params['SRC_STRING']}:{paths['CUSTOMIZED_TOK']}.{params['CODES']}.pth\;{params['TGT_STRING']}:")
    
    elif params['BILINGUAL_DICT_SUP']:
        print('Unsupervised training with supervised bilingual dictionnary training')
        
        # Call main.py
        os.path(f"python main.py \
                  --exp_name {params['EXP_NAME']}_BD \
                  --batch_size 16 \
                  --wandb \
                  --max_epoch 40 \
                  --transformer {params['TRANSFORMER']} \
                  --n_enc_layers 4 \
                  --n_dec_layers 4 \
                  --share_enc 3 \
                  --share_dec 3 \
                  --share_lang_emb True \
                  --share_output_emb True \
                  --langs {params['SRC_STRING']},{params['CODES']} \
                  --n_mono -1 \
                  --mono_dataset {params['SRC_STRING']}:{paths['EXP']}/{params['SRC_STRING']}.tok.{params['CODES']}.pth,,\;\
                                 {params['TGT_STRING']}:{paths['EXP']}/{params['TGT_STRING']}.tok.{params['CODES']}.pth,,' \
                  --n_para -1 \
                  --para_dataset {params['SRC_STRING']}-{params['TGT_STRING']}:{paths['BILINGUAL_DICT']}/XX.tok.{params['CODES']}.pth,\
                                                                               {paths['EXP']}/XX_VALID.tok.{params['CODES']}.pth,\
                                                                               {paths['EXP']}/XX_TEST.tok.{params['CODES']}.pth \
                  --mono_directions {params['SRC_STRING']},{params['CODES']} \
                  --para_directions {params['SRC_STRING']}-{params['CODES']},{params['CODES']}-{params['SRC_STRING']} \
                  --word_shuffle 3 \
                  --word_dropout 0.1 \
                  --word_blank 0.2 \
                  --pivo_directions {params['TGT_STRING']}-{params['SRC_STRING']}-{params['TGT_STRING']},{params['SRC_STRING']}-{params['TGT_STRING']}-{params['SRC_STRING']} \
                  --pretrained_emb {paths['CONCAT_BPE']}.vec \
                  --pretrained_out True \
                  --lambda_xe_mono '0:1,100000:0.1,300000:0' \
                  --lambda_xe_para '0:1,100000:0.1,300000:0' \
                  --lambda_xe_otfd 1 \
                  --otf_num_processes 30 \
                  --otf_sync_params_every 1000 \
                  --enc_optimizer adam,lr=0.0001 \
                  --epoch_size 100000 \
                  --stopping_criterion 'bleu_{params['SRC_STRING']}_{params['TGT_STRING']}_valid,20' \
                  --customized_data {params['SRC_STRING']}:{paths['CUSTOMIZED_TOK']}.{params['CODES']}.pth\;{params['TGT_STRING']}:")

    elif params['PROP_SUPERVISED'] == 0.:
        print('Unsupervised Training!') 
        
        os.system(f"python main.py\
                   --exp_name {params['EXP_NAME']} \
                   --batch_size 16 \
                   --wandb \
                   --max_epoch 40 \
                   --transformer {params['TRANSFORMER']} \
                   --n_enc_layers 4 \
                   --n_dec_layers 4 \
                   --share_enc 3 \
                   --share_dec 3 \
                   --share_lang_emb True \
                   --share_output_emb True \
                   --langs {params['SRC_STRING']},{params['TGT_STRING']} \
                   --n_mono -1 \
                   --mono_dataset {params['SRC_STRING']}:{paths['EXP']}/{params['SRC_STRING']}.tok.{params['CODES']}.pth,,\;{params['TGT_STRING']}:{paths['EXP']}/{params['TGT_STRING']}.tok.{params['CODES']}.pth,,\
                   --para_dataset {params['SRC_STRING']}-{params['TGT_STRING']}:,{paths['EXP']}/XX_VALID.tok.{params['CODES']}.pth,{paths['EXP']}/XX_TEST.tok.{params['CODES']}.pth \
                   --mono_directions {params['SRC_STRING']},{params['TGT_STRING']} \
                   --word_shuffle 3 \
                   --word_dropout 0.1 \
                   --word_blank 0.2 \
                   --pivo_directions {params['TGT_STRING']}-{params['SRC_STRING']}-{params['TGT_STRING']},{params['SRC_STRING']}-{params['TGT_STRING']}-{params['SRC_STRING']} \
                   --pretrained_emb {paths['CONCAT_BPE']}.vec \
                   --pretrained_out True \
                   --lambda_xe_mono '0:1,100000:0.1,300000:0' \
                   --lambda_xe_otfd 1 \
                   --otf_num_processes 30 \
                   --otf_sync_params_every 1000 \
                   --enc_optimizer adam,lr=0.0001 \
                   --epoch_size 100000 \
                   --stopping_criterion 'bleu_{params['SRC_STRING']}_{params['TGT_STRING']}_valid,20' \
                   {custom_file_arg}")
        
    else:
        print(f"Training with a proportion of {params['PROP_SUPERVISED']} supervised examples !")
        """
        EXP_NAME=$EXP_NAME'_'$SUPERVISED'_Prop_Supervised'
        N_SUP_EX=int(100000*params['PROP_SUPERVISED'])

        echo "Creating target .pth..."
        # WE NEED TO DO ALL THAT BECAUSE WE NEED TO TOKENIZE WITH THE SAME BPE
        mkdir -p ./data/artificial_grammars/$DATA_FOLDER/SUP
        
        TGT_RAW=$GRAMMARS_PATH/$DATA_FOLDER'_SUP'/target/sample_$TGT_NAME.txt

        echo "TGT_RAW is $TGT_RAW"
        TGT_TOK=$GRAMMARS_PATH/$DATA_FOLDER/SUP/sample_t$TGT_NAME.LEX$LEXICON_TGT.tok
        
        # tokenize data
        if ! [[ -f "$TGT_TOK" ]]; then
            echo "Tokenizing data..."
            cat $TGT_RAW | $NORM_PUNC -l en | $TOKENIZER -l en -no-escape -threads $N_THREADS > $TGT_TOK
        fi
        
        # apply fastbpe
        echo "Applying FastBPE..."
        $FASTBPE applybpe $TGT_TOK.$CODES $TGT_TOK $BPE_CODES $TGT_VOCAB
        
        # Binarizing data...
        echo "Binarizing data..."
        rm -f $TGT_TOK.$CODES.pth
        $UMT_PATH/preprocess.py $FULL_VOCAB $TGT_TOK.$CODES
        
        # Copy source .pth and adding SUP in its name
        os.system(f"cp {paths['EXP']}/{params['SRC_STRING']}.tok.{params['CODES']}.pth {paths['EXP']}/{params['SRC_STRING']}_SUP.tok.{params['CODES']}.pth)
        
        print("Done! Running main.py...")
        
        # Call main.py
        os.system(f"python main.py \
                   --exp_name {params['EXP_NAME']} \
                   --batch_size 16 \
                   --wandb \
                   --max_epoch 40 \
                   --transformer {params['TRANSFORMER']} \
                   --n_enc_layers 4 \
                   --n_dec_layers 4 \
                   --share_enc 3 \
                   --share_dec 3 \
                   --share_lang_emb True \
                   --share_output_emb True \
                   --langs {params['SRC_STRING']},{params['TGT_STRING']} \
                   --n_mono -1 \
                   --mono_dataset {params['SRC_STRING']}:{paths['EXP']}/{params['SRC_STRING']}.tok.{params['CODES']}.pth,,;\
                                  {params['TGT_STRING']}:{paths['EXP']}/{params['TGT_STRING']}.tok.{params['CODES']}.pth,,' \
                   --n_para {N_SUP_EX} \
                   --para_dataset {params['SRC_STRING']}-{params['TGT_STRING']}:{paths['EXP']}/XX_SUP.tok.{params['CODES']}'.pth,\
                                                                                {paths['EXP']}/XX_VALID.tok.{params['CODES']}.pth,\
                                                                                {paths['EXP']}/XX_TEST.tok.{params['CODES']}.pth' \
                   --mono_directions {params['SRC_STRING']},{params['TGT_STRING']}  \
                   --para_directions {params['SRC_STRING']}-{params['TGT_STRING']} ,{params['TGT_STRING']}-{params['SRC_STRING']} \
                   --word_shuffle 3 \
                   --word_dropout 0.1 \
                   --word_blank 0.2 \
                   --pivo_directions {params['TGT_STRING']}-{params['SRC_STRING']}-{params['TGT_STRING']},{params['SRC_STRING']}-{params['TGT_STRING']}-{params['SRC_STRING']} \
                   --pretrained_emb {paths['CONCAT_BPE']}.vec \
                   --pretrained_out True \
                   --lambda_xe_mono '0:1,100000:0.1,300000:0' \
                   --lambda_xe_para '0:1,100000:0.1,300000:0' \
                   --lambda_xe_otfd 1 \
                   --otf_num_processes 30 \
                   --otf_sync_params_every 1000 \
                   --enc_optimizer adam,lr=0.0001 \
                   --epoch_size 100000 \
                   --stopping_criterion 'bleu_{params['SRC_STRING']}_{params['TGT_STRING']}_valid,20' \
                   --customized_data {params['SRC_STRING']}:{paths['CUSTOMIZED_TOK']}.{params['CODES']}.pth;{params['TGT_STRING']}:")
        """