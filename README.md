
This code belongs to the implementation of <a href = "http://jmlr.org/proceedings/papers/v48/trouillon16.pdf">"Complex Embeddings for Simple Link Prediction"</a> paper.

<h2>Requirements</h2>

<ul>
<li>Python 3 </li>
<li>Tensorflow > 0.8</li>
<li>Numpy</li>
</ul>

<h2>Training</h2>

Print parameters:
<pre>
./train.py --help
</pre>
<pre>
optional arguments:
  -h, --help            show this help message and exit
  --embedding_dim EMBEDDING_DIM
                        Dimensionality of character embedding (default: 128)
  --l2_reg_lambda L2_REG_LAMBDA
                        L2 regularizaion lambda (default: 0.0)
  --dropout_keep_prob DROPOUT_KEEP_PROB
                        Dropout keep probability (default: 0.5)
  --batch_size BATCH_SIZE
                        Batch Size (default: 64)
  --num_epochs NUM_EPOCHS
                        Number of training epochs (default: 100)
  --evaluate_every EVALUATE_EVERY
                        Evaluate model on dev set after this many steps
                        (default: 100)
  --checkpoint_every CHECKPOINT_EVERY
                        Save model after this many steps (default: 100)
  --allow_soft_placement ALLOW_SOFT_PLACEMENT
                        Allow device soft device placement
  --noallow_soft_placement
  --log_device_placement LOG_DEVICE_PLACEMENT
                        Log placement of ops on devices
  --nolog_device_placement
</pre>
Train:
<pre>
./train.py
</pre>

<h2>Link Predicting</h2>
<pre>

./link_prediction.py --checkpoint_dir="./runs/1451237919/checkpoints/"
</pre>

Replace the checkpoint dir with the output from the training. To use your own data, change the eval.py script to load your data.

