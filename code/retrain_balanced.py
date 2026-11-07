import os, numpy as np, nibabel as nib, tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from pathlib import Path
from PIL import Image
from sklearn.metrics import f1_score, precision_score, recall_score, confusion_matrix, accuracy_score

print('TensorFlow:', tf.__version__)
TRAINING_DIR = Path('training_data_v2')
IMG_SIZE = 256
EPOCHS = 25
BATCH_SIZE = 8
TRAIN_CASES = list(range(0, 17))
TEST_CASES = [20, 25, 30]

def dice_coef(y_true, y_pred, smooth=1e-6):
    y_true_f = tf.keras.backend.flatten(tf.cast(y_true, tf.float32))
    y_pred_f = tf.keras.backend.flatten(tf.cast(y_pred, tf.float32))
    intersection = tf.keras.backend.sum(y_true_f * y_pred_f)
    return (2.*intersection+smooth)/(tf.keras.backend.sum(y_true_f)+tf.keras.backend.sum(y_pred_f)+smooth)

def weighted_bce_dice_loss(y_true, y_pred, pos_weight=43.94):
    y_true = tf.cast(y_true, tf.float32)
    y_pred = tf.cast(y_pred, tf.float32)
    
    # Flatten everything
    y_true_flat = tf.reshape(y_true, [-1])
    y_pred_flat = tf.reshape(y_pred, [-1])
    
    # Binary crossentropy per pixel
    bce = tf.keras.losses.binary_crossentropy(y_true_flat, y_pred_flat)
    
    # Weight positive class more
    weights = 1.0 + y_true_flat * (pos_weight - 1.0)
    weighted_bce = tf.reduce_mean(bce * weights)
    
    # Dice loss
    dice_loss = 1.0 - dice_coef(y_true, y_pred)
    
    return weighted_bce + dice_loss

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

def load_slices(cid, max_slices=20):
    vp=TRAINING_DIR/f'volume-{cid}.nii'
    sp=TRAINING_DIR/f'segmentation-{cid}.nii'
    if not vp.exists(): return [],[]
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
    print(f'Case {cid}: {len(X)} slices'); return X,Y

print('\nLoading TRAINING cases (17 cases)...')
train_X, train_Y = [], []
for cid in TRAIN_CASES:
    X,Y = load_slices(cid)
    for img,msk in zip(X,Y):
        for ax,ay in [(img,msk),(np.fliplr(img),np.fliplr(msk)),(np.flipud(img),np.flipud(msk)),(np.rot90(img),np.rot90(msk))]:
            train_X.append(ax); train_Y.append(ay)

print('\nLoading TEST cases (3 unseen)...')
test_X, test_Y = [], []
for cid in TEST_CASES:
    X,Y = load_slices(cid)
    test_X.extend(X); test_Y.extend(Y)

Xa=np.array(train_X)[...,np.newaxis]; Ya=np.array(train_Y)[...,np.newaxis]
Xtest=np.array(test_X)[...,np.newaxis]; Ytest=np.array(test_Y)[...,np.newaxis]
idx=np.random.permutation(len(Xa)); Xa=Xa[idx]; Ya=Ya[idx]
sp=int(len(Xa)*0.8)
Xt,Xv,Yt,Yv=Xa[:sp],Xa[sp:],Ya[:sp],Ya[sp:]

bg_count = np.sum(Yt == 0)
liver_count = np.sum(Yt == 1)
pos_weight = bg_count / (liver_count + 1e-6)

print(f'\nClass Balance:')
print(f'  Background: {bg_count:,}')
print(f'  Liver: {liver_count:,}')
print(f'  Weight ratio: {pos_weight:.2f}x')

model=build_unet(IMG_SIZE)
model.compile(optimizer=keras.optimizers.Adam(1e-4),
    loss=lambda yt,yp: weighted_bce_dice_loss(yt,yp,pos_weight),
    metrics=[dice_coef])

cb=[
    keras.callbacks.ModelCheckpoint('best_unet_lits17_balanced.keras',monitor='val_dice_coef',mode='max',save_best_only=True,verbose=1),
    keras.callbacks.EarlyStopping(monitor='val_dice_coef',mode='max',patience=5,verbose=1)
]

print('\nTraining WITH WEIGHTED LOSS...')
history=model.fit(Xt,Yt,validation_data=(Xv,Yv),epochs=EPOCHS,batch_size=BATCH_SIZE,callbacks=cb,verbose=1)

model=tf.keras.models.load_model('best_unet_lits17_balanced.keras',
    custom_objects={'dice_coef':dice_coef})

print('\n' + '='*70)
print('TEST SET RESULTS: WITH WEIGHTED LOSS (FIXED IMBALANCE)')
print('='*70)

preds = model.predict(Xtest, verbose=0)
preds_binary = (preds > 0.5).astype(np.uint8)
y_true_flat = Ytest.flatten().astype(np.uint8)
y_pred_flat = preds_binary.flatten()

accuracy = accuracy_score(y_true_flat, y_pred_flat)
f1 = f1_score(y_true_flat, y_pred_flat, zero_division=0)
precision = precision_score(y_true_flat, y_pred_flat, zero_division=0)
recall = recall_score(y_true_flat, y_pred_flat, zero_division=0)
cm = confusion_matrix(y_true_flat, y_pred_flat)
tn, fp, fn, tp = cm.ravel()
sensitivity = tp/(tp+fn)
specificity = tn/(tn+fp)

smooth=1e-6
intersection = (y_true_flat * y_pred_flat).sum()
dice = (2.*intersection+smooth)/(y_true_flat.sum()+y_pred_flat.sum()+smooth)

print(f'\nAccuracy:    {accuracy:.4f}')
print(f'Dice:        {dice:.4f}')
print(f'F1 Score:    {f1:.4f}')
print(f'Precision:   {precision:.4f}')
print(f'Recall:      {recall:.4f}  ← EXPECT 0.60-0.75 (was 0.24)')
print(f'Sensitivity: {sensitivity:.4f}')
print(f'Specificity: {specificity:.4f}')
print(f'\nConfusion Matrix:')
print(f'  TN: {tn:>12,}  FP: {fp:>12,}')
print(f'  FN: {fn:>12,}  TP: {tp:>12,}')
print(f'\nModel saved: best_unet_lits17_balanced.keras')
