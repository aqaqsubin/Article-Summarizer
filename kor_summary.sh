#!/bin/bash
data="articles/"
origin=$data"Origin-Data/"
preprocessed=$data"Preprocessed-Data/"

summary_preprocessed=$data"Summary-Preprocessed-Data/"
title_preprocessed=$data"Title-Preprocessed-Data/"

valid=$data"Valid-Data/"
valid_preprocessed=$data"Valid-Preprocessed-Data/"

valid_summary_preprocessed=$data"Valid-Summary-Preprocessed-Data/"
valid_title_preprocessed=$data"Valid-Title-Preprocessed-Data/"

src_dir="src/"
data_src_dir=$src_dir"data_preprocessing/"
model_dir=$src_dir"trained-model/"
train_dir=$src_dir"train/"
eval_dir=$src_dir"evaluate/"

word_encoding_model=$data_src_dir"Word-Encoding-Model/"

{   
    ## 데이터 전처리 (전처리된 데이터 이미 있는지 확인)
    if test -d $preprocessed; then
        echo "Already Exist Preprocessed Data."
    else 
        echo "Data Preprocessing"

        # 불용어, 특수문자 제거 등 데이터 전처리 
        python src/data_Preprocessing/dataPreprocessor.py --tokenizer=False

        # 전처리 된 데이터를 바탕으로 Word Encoding Model 생성
        if test -f $word_encoding_model; then
            echo "Already Exist Word Encoding Model"
        else 
            python src/data_Preprocessing/wordEncoding.py
        fi

        # Word Encoding Model에 따른 토큰화 및 입력 시퀀스 길이 정제
        python src/data_Preprocessing/dataPreprocessor.py --tokenizer=True
    fi

    transformer_model=$model_dir"Transformer/"
    seq2seq_model=$model_dir"Seq2Seq/"

    # 프로세스 선택 (제목 생성/요약문 생성)
    echo "1: Generate Headline"
    echo "2: Generate Summary"
    echo -n "Select Process : "
    read process

    [[ $process = 1 ]] && process=true || process=false

    # 모델 선택
    echo "1: Transformer"
    echo "2: Transformer + RC-Encoder"
    echo "3: Sequence-to-Sequence"

    echo -n "Select Model : "
    read id
    case $id in
        "1") model=$transformer_model
        model_src=$train_dir"train_transformer.py"
        model_option="--headline=$process"
        ;;
        "2") echo -n "Input N : "
        read N
        trans_rc_enc_N_model=$model_dir"Transformer_RC_Encoder_$N"
        model=$trans_rc_enc_N_model
        model_src=$train_dir"train_trans_rc_enc.py"
        model_option="--headline=$process --n=$N"
        ;;
        "3") model=$seq2seq_model
        model_src=$train_dir"train_seq2seq.py"
        model_option="--headline=$process"
        ;;
        *) echo "Invalid Model";;
    esac

    ## 모델 훈련 (모델 체크포인트 있는지 확인)
    echo $model
    if test -d $model; then
        echo "Already Exist Trained Model."
    else 
        echo "Train Model"
        python $model_src $model_option
    fi

    ## 모델 검증
    case $id in
        "1")
        eval_src=$eval_dir"eval_transformer.py"
        ;;
        "2") 
        eval_src=$eval_dir"eval_trans_rc_enc.py"
        ;;
        "3") 
        eval_src=$eval_dir"eval_seq2seq.py"
        ;;

    esac
    echo "Evaluate Model"
    python $eval_src $model_option

    echo "finished"
} || {
    
    echo "An unexpected exception was thrown"
    throw $ex_code 
}