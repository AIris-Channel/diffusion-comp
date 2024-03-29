import ml_collections


def d(**kwargs):
    """Helper of creating a config dict."""
    return ml_collections.ConfigDict(initial_dictionary=kwargs)


def get_config():
    config = ml_collections.ConfigDict()

    config.seed = 1234
    config.pred = 'noise_pred'
    config.z_shape = (4, 64, 64)
    config.clip_img_dim = 512
    config.clip_text_dim = 768
    config.text_dim = 64  # reduce dimension
    config.data_type = 1
    config.gradient_accumulation_steps = 1
    config.log_interval = 10
    config.eval_interval = 1000
    config.save_interval = 1000
    config.save_best = True
    config.max_step = 5000

    config.num_workers = 1
    config.batch_size = 2
    config.resolution = 512

    config.clip_img_model = "ViT-B/32"
    config.clip_text_model = "openai/clip-vit-large-patch14"

    config.only_load_model = True

    config.use_blip_caption = False
    
    config.save_target_key = 'image_proj_model'

    config.optimizer = d(
        name='adamw',
        lr=1e-5,
        weight_decay=0.03,
        betas=(0.9, 0.9),
        amsgrad=False
    )
    # config.optimizer = d(
    #     name = 'lion',
    #     lr = 2e-5,
    # )

    config.lr_scheduler = d(
        name='customized',
        warmup_steps=20
    )

    config.autoencoder = d(
        pretrained_path='models/autoencoder_kl.pth',
    )

    config.caption_decoder = d(
        pretrained_path="models/caption_decoder.pth",
        hidden_dim=config.get_ref('text_dim'),
        tokenizer_path="./models/gpt2"
    )

    config.nnet = d(
        img_size=64,
        in_chans=4,
        patch_size=2,
        embed_dim=1536,
        depth=30,
        num_heads=24,
        mlp_ratio=4,
        qkv_bias=False,
        pos_drop_rate=0.,
        drop_rate=0.,
        attn_drop_rate=0.,
        mlp_time_embed=False,
        text_dim=config.get_ref('text_dim'),
        num_text_tokens=77,
        clip_img_dim=config.get_ref('clip_img_dim'),
        use_checkpoint=False
    )

    # sample
    config.mode = "t2i"
    config.n_samples = 4
    config.n_iter = 1
    config.nrow = 4
    config.sample = d(
        sample_steps=30,
        scale=7.,
        t2i_cfg_mode='true_uncond'
    )
    
    # lora
    config.lora_dim = 8
    config.lora_alpha = 4
    config.lora_dropout = 0.05
    config.train_text_encoder = True
    config.text_encoder_lr = 2e-5
    config.train_nnet = True
    config.nnet_lr = 2e-5
    
    # text inversion
    config.token_string = 'mychar'
    config.init_word = 'highly detailed'
    config.num_vectors_per_token = 10
    
    
    # adversarial training
    config.use_discriminator = False
    config.disc_steps = 30
    config.disc_loss_weight = 1e-2

    # ip-adapter
    config.image_proj_tokens = 4
    
    return config
