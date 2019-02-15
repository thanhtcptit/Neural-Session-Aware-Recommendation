# ------- Lastfm --------

# Concatenate - simple
# python main.py --mode=train --name=UserGRU-pre --input=concat --fusion_type=pre --train_file=clean-lastfm-train --test_file=clean-lastfm-dev;
# python main.py --mode=train --name=UserGRU-post --input=concat --fusion_type=post --train_file=clean-lastfm-train --test_file=clean-lastfm-dev;

# Concatenate - attention
# python main.py --mode=train --name=UserAGRU-pre --input=attention --fusion_type=pre --train_file=clean-lastfm-train --test_file=clean-lastfm-dev;
# python main.py --mode=train --name=UserAGRU-post --input=attention --fusion_type=post --train_file=clean-lastfm-train --test_file=clean-lastfm-dev;

# Concatenate - attention fixed
# python main.py --mode=train --name=UserAGGRU-pre --input=attention-global --fusion_type=pre --train_file=clean-lastfm-train --test_file=clean-lastfm-dev;
# python main.py --mode=train --name=UserAGGRU-post --input=attention-global --fusion_type=post --train_file=clean-lastfm-train --test_file=clean-lastfm-dev;

# Sum - simple
#python3 main.py --mode=train --name=UserGRU-sum-pre --input=sum --fusion_type=pre --train_file=clean-avito-train --test_file=clean-avito-dev;
#python3 main.py --mode=train --name=UserGRU-sum-post --input=sum --train_file=clean-avito-train --test_file=clean-avito-dev --num_epoch=6;

# Sum - attention
#python3 main.py --mode=train --name=UserAGRU-sum-pre --input=attention-sum --fusion_type=pre --train_file=clean-avito-train --test_file=clean-avito-dev;

# Sum - attention fixed
# python3 main.py --mode=train --name=UserAGGRU-sum-pre --input=attention-fixed-sum --fusion_type=pre --train_file=clean-avito-train --test_file=clean-avito-dev;
# python3 main.py --mode=train --name=UserAGGRU-sum-post --input=attention-fixed-sum --train_file=clean-lastfm-train --test_file=clean-lastfm-dev --num_epoch=6;

# Mul - simple
# python main.py --mode=train --name=UserGRU-mul-pre --input=mul --fusion_type=pre --train_file=clean-lastfm-train --test_file=clean-lastfm-dev;
# python main.py --mode=train --name=UserGRU-mul-post --input=mul --train_file=clean-lastfm-train --test_file=clean-lastfm-dev;

# CF
# python3 main.py --mode=train --name=UserCF-concat --input=concat --fusion_type=cf --train_file=clean-lastfm-train --test_file=clean-lastfm-dev;
# python3 main.py --mode=train --name=UserCF-sum --input=sum --fusion_type=cf --train_file=clean-lastfm-train --test_file=clean-lastfm-dev;
# python3 main.py --mode=train --name=UserCF-mul --input=mul --fusion_type=cf --train_file=clean-lastfm-train --test_file=clean-lastfm-dev;
python3 main.py --mode=train --name=UserAGRU-cf --input=attention --fusion_type=cf --train_file=clean-lastfm-train --test_file=clean-lastfm-dev;
python3 main.py --mode=train --name=UserAGGRU-cf --input=attention-global --fusion_type=cf --train_file=clean-lastfm-train --test_file=clean-lastfm-dev;
python3 main.py --mode=train --name=UserAGRU-sum-cf --input=attention-sum --fusion_type=cf --train_file=clean-lastfm-train --test_file=clean-lastfm-dev;
python3 main.py --mode=train --name=UserAGGRU-sum-cf --input=attention-fixed-sum --fusion_type=cf --train_file=clean-lastfm-train --test_file=clean-lastfm-dev;
