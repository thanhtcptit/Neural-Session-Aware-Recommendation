# python main.py --mode=train --name=UserGru-mul-avito --input=mul --train_file=clean-avito-train --test_file=clean-avito-dev;
# python main.py --mode=train --name=UserGru-mul-ff-avito --input=mul-ff --train_file=clean-avito-train --test_file=clean-avito-dev;
# python main.py --mode=train --name=UserGru-cf-avito --input=cf --train_file=clean-avito-train --test_file=clean-avito-dev;
# python main.py --mode=train --name=UserAGru-avito --input=attention --train_file=clean-avito-train --test_file=clean-avito-dev;
# python main.py --mode=train --name=UserAGru-global-avito --input=attention-global --train_file=clean-avito-train --test_file=clean-avito-dev;
# python main.py --mode=train --name=UserGru-pre-concat-avito --input=concat --train_file=clean-avito-train --test_file=clean-avito-dev --fusion_type=pre;
#python main.py --mode=train --name=UserAGru-ew-avito --input=attention-ew --train_file=clean-avito-train --test_file=clean-avito-dev;
#python main.py --mode=train --name=UserAGru-sum-avito --input=attention-sum --train_file=clean-avito-train --test_file=clean-avito-dev;

# Lastfm
# Concatenate - simple
# python main.py --mode=train --name=UserGRU-pre --input=concat --fusion_type=pre --train_file=clean-lastfm-train --test_file=clean-lastfm-dev;
# Concatenate - attention
# python main.py --mode=train --name=UserAGRU-pre --input=attention --fusion_type=pre --train_file=clean-lastfm-train --test_file=clean-lastfm-dev;
# Concatenate - attention fixed
# python main.py --mode=train --name=UserAGGRU-pre --input=attention-global --fusion_type=pre --train_file=clean-lastfm-train --test_file=clean-lastfm-dev;
# Mul - simple
# python main.py --mode=train --name=UserGRU-mul-pre --input=mul --fusion_type=pre --train_file=clean-lastfm-train --test_file=clean-lastfm-dev;
# python main.py --mode=train --name=UserGRU-mul-post --input=mul --train_file=clean-lastfm-train --test_file=clean-lastfm-dev;
# Sum - simple
python main.py --mode=train --name=UserGRU-sum-pre --input=sum --fusion_type=pre --train_file=clean-lastfm-train --test_file=clean-lastfm-dev;
python main.py --mode=train --name=UserGRU-sum-post --input=sum --train_file=clean-lastfm-train --test_file=clean-lastfm-dev;
# Sum - attention
python main.py --mode=train --name=UserAGRU-sum-pre --input=attention-sum --fusion_type=pre --train_file=clean-lastfm-train --test_file=clean-lastfm-dev;
# Sum - attention fixed
python main.py --mode=train --name=UserAGGRU-sum-pre --input=attention-fixed-sum --fusion_type=pre --train_file=clean-lastfm-train --test_file=clean-lastfm-dev;
python main.py --mode=train --name=UserAGGRU-sum-post --input=attention-fixed-sum --train_file=clean-lastfm-train --test_file=clean-lastfm-dev;