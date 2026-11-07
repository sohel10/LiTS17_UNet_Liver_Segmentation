import os, numpy as np, nibabel as nib, tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from pathlib import Path
from PIL import Image
from sklearn.metrics import f1_score, precision_score, recall_score, confusion_matrix

print('TensorFlow:', tf.__version__)
TRAINING_DIR = Path('training_data')
IMG_SIZE = 256
EPOCHS = 20
BATCH_SIZE = 4
TRAIN_CASES = [77, 31, 45, 66, 72, 53, 54]   # 7 for training
TEST_CASES  = [63, 37, 41]                     # 3 for testing (unseen!)

def dice_coef(y_true, y_pred, smooth=1e-6):
    y_true_f = tf.keras.backend.flatten(tf.cast(y_true, tf.float32))
    y_pred_f = tf.keras.backend.flatten(tf.cast(y_pred, tf.float32))
    intersection = tf.keras.backend.sum(y_true_f * y_pred_f)
    return (2.*intersection+smooth)/(tf.keras.backend.sum(y_true_f)+tf.keras.backend.sum(y_pred_f)+smooth)

def bce_dice_loss(y_true, y_pred):
    return tf.keras.losses.binary_crossentropy(y_true,y_pred)+(1-dice_coef(y_true,y_pred))

def conv_block(x, f):
    x = layers.Conv2D(f,3,padding='same',activation='relu')(x)
    x = layers.Conv2D(f,3,padding='same',activation='relu')(x)
    return x

def build_unet(sz=256):
    inp = keras.Input(shape=(sz,sz,1))
    c1=conv_block(inp,32); p1=layers.MaxPooling2D()(c1)
    c2=conv_block(p1,64);  p2=layers.MaxPooling2D()(c2)
    c3=conv_block(p2,128); p3=layers.MaxPooling2D()(c3)
    c4=conv_block(p3,256)
    u5=layers.UpSampling2D()(c4); u5=layers.Concatenate()([u5,c3]); c5=conv_block(u5,128)
    u6=layers.UpSampling2D()(c5); u6=layers.Concatenate()([u6,c2]); c6=conv_block(u6,64)
    u7=layers.UpSampling2D()(c6); u7=layers.Concatenate()([u7,c1]); c7=conv_block(u7,32)
    out=layers.Conv2D(1,1,activation='sigmoid')(c7)
    return keras.Model(inp,out)

def load_slices(cid, max_slices=15):
    vp=TRAINING_DIR/f'volume-{cid}.nii'
    sp=TRAINING_DIR/f'segmentation-{cid}.nii'
    if not vp.exists(): print(f'Skip {cid}'); return [],[]
    vol=nib.load(str(vp)).get_fdata()
    seg=(nib.load(str(sp)).get_fdata()>=1).astype(np.float32)
    X,Y=[],[]
    for i in range(vol.shape[2]):
        if seg[:,:,i].sum()<50: continue
        v=vol[:,:,i].astype(np.float32)
        mn,mx=v.min(),v.max()
        if mx>mn: v=(v-mn)/(mx-mn)
        vi=np.array(Image.fromarray(v).resize((IMG_SIZE,IMG_SIZE),Image.BILINEAR),dtype=np.float32)
        si=np.array(Image.fromarray(seg[:,:,i]).resize((IMG_SIZE,IMG_SIZE),Image.NEAREST),dtype=np.float32)
        X.append(vi); Y.append(si)
        if len(X)>=max_slices: break
    print(f'Case {cid}: {len(X)} slices')
    return X,Y

# ── Load TRAINING data with augmentation ───────────────────────────────────
print('\nLoading TRAINING cases...')
train_X, train_Y = [], []
for cid in TRAIN_CASES:
    X,Y = load_slices(cid)
    for img,msk in zip(X,Y):
        for ax,ay in [(img,msk),(np.fliplr(img),np.fliplr(msk)),(np.flipud(img),np.flipud(msk)),(np.rot90(img),np.rot90(msk))]:
            train_X.append(ax); train_Y.append(ay)

# ── Load TEST data WITHOUT augmentation ────────────────────────────────────
print('\nLoading TEST cases (unseen data)...')
test_X, test_Y = [], []
for cid in TEST_CASES:
    X,Y = load_slices(cid)
    test_X.extend(X); test_Y.extend(Y)

print(f'\nTraining samples: {len(train_X)}')
print(f'Test samples:     {len(test_X)}')

# Convert to arrays
Xa=np.array(train_X)[...,np.newaxis]; Ya=np.array(train_Y)[...,np.newaxis]
Xtest=np.array(test_X)[...,np.newaxis]; Ytest=np.array(test_Y)[...,np.newaxis]

# Shuffle training only
idx=np.random.permutation(len(Xa)); Xa=Xa[idx]; Ya=Ya[idx]

# Validation split from training (80/20)
sp=int(len(Xa)*0.8)
Xt,Xv,Yt,Yv=Xa[:sp],Xa[sp:],Ya[:sp],Ya[sp:]
print(f'Train:{len(Xt)} | Val:{len(Xv)} | Test:{len(Xtest)}')

# ── Build and train model ───────────────────────────────────────────────────
model=build_unet(IMG_SIZE)
model.compile(
    optimizer=keras.optimizers.Adam(1e-4),
    loss=bce_dice_loss,
    metrics=[dice_coef]
)

cb=[
    keras.callbacks.ModelCheckpoint('best_unet_lits17_v2.keras',monitor='val_dice_coef',mode='max',save_best_only=True,verbose=1),
    keras.callbacks.EarlyStopping(monitor='val_dice_coef',mode='max',patience=5,verbose=1)
]

history=model.fit(Xt,Yt,validation_data=(Xv,Yv),epochs=EPOCHS,batch_size=BATCH_SIZE,callbacks=cb,verbose=1)

# ── Evaluate on TEST set (unseen data) ─────────────────────────────────────
print('\n' + '='*50)
print('EVALUATION ON UNSEEN TEST SET')
print('='*50)

model.load_weights('best_unet_lits17_v2.keras')
preds = model.predict(Xtest, verbose=0)
preds_binary = (preds > 0.5).astype(np.uint8)

# Flatten for sklearn metrics
y_true_flat = Ytest.flatten().astype(np.uint8)
y_pred_flat = preds_binary.flatten()

# Calculate all metrics
f1        = f1_score(y_true_flat, y_pred_flat, zero_division=0)
precision = precision_score(y_true_flat, y_pred_flat, zero_division=0)
recall    = recall_score(y_true_flat, y_pred_flat, zero_division=0)
cm        = confusion_matrix(y_true_flat, y_pred_flat)

# Dice on test set
smooth = 1e-6
intersection = (y_true_flat * y_pred_flat).sum()
dice = (2. * intersection + smooth) / (y_true_flat.sum() + y_pred_flat.sum() + smooth)

print(f'\nTest Results:')
print(f'  Dice Score:  {dice:.4f}')
print(f'  F1 Score:    {f1:.4f}')
print(f'  Precision:   {precision:.4f}')
print(f'  Recall:      {recall:.4f}')
print(f'\nConfusion Matrix:')
print(f'  TN: {cm[0,0]:>8,}  FP: {cm[0,1]:>8,}')
print(f'  FN: {cm[1,0]:>8,}  TP: {cm[1,1]:>8,}')
print(f'\nInterpretation:')
print(f'  TP = Correctly identified liver pixels')
print(f'  TN = Correctly identified background pixels')
print(f'  FP = Background wrongly predicted as liver')
print(f'  FN = Liver pixels missed by model')
print(f'\nModel saved: best_unet_lits17_v2.keras')
print('Training complete!')
