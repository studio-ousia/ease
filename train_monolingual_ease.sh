python main.py \
    model_args.model_name_or_path=bert-base-uncased \
    model_args.ease_temp=100 \
    model_args.ease_loss_ratio=0.01 \
    model_args.pooler_type=cls \
    data_args.dataset_name_or_path=wiki_en \
    train_args.learning_rate=3e-05 \
    train_args.output_dir=result/my-ease-bert-base-uncased \
    train_args.per_device_train_batch_size=8 \
    train_args.gradient_accumulation_steps=16
