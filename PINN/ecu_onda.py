import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
import itertools as it
import tensorflow as tf
from matplotlib.animation import FuncAnimation
import time

@tf.function
def edo_func(modelo,feature):
  phi=modelo(feature)
  # x
  dphi=tf.gradients(phi,feature)[0]


  ddx_phi=tf.gradients(dphi[0],feature)[0]
  ddy_phi=tf.gradients(dphi[1],feature)[0]
  ddt_phi=tf.gradients(dphi[2],feature)[0]

  c=10.0


  zeta=ddt_phi-c*((ddx_phi)+(ddy_phi))
  return zeta


def condiciones(modelo,feature_i,phi_r):
  norte=feature_i[:,[1,6,0]]
  sur=feature_i[:,[1,4,0]]
  este=feature_i[:,[5,2,0]]
  oeste=feature_i[:,[3,2,0]]

  centro=np.array([10.0,10.0,0.0])[None,:]

  norte_mol=modelo(norte)
  sur_mol=modelo(sur)
  este_mol=modelo(este)
  oeste_mol=modelo(oeste)

  centro_mol=modelo(centro)

  ini_real=phi_r

  err=tf.reduce_mean(tf.square([norte_mol - ini_real, sur_mol - ini_real ,este_mol - ini_real ,oeste_mol - ini_real]))
  err+=tf.reduce_mean(tf.square(centro_mol-4.0))
  return err
#funcion de coste
def perdida(modelo,feature,feature_i,phi_r):
  zeta_nau=edo_func(modelo,feature)
  zeta_inicial=condiciones(modelo,feature_i,phi_r)
  return tf.reduce_mean(tf.square(zeta_nau))+zeta_inicial

#gradietne de la perdida
def perdida_gradiente(modelo,feature,feature_i,phi_r):
  with tf.GradientTape() as tape:
    loss=perdida(modelo,feature,feature_i,phi_r)
  grand=tape.gradient(loss,modelo.trainable_weights)
  return [loss,grand]




def optimizar(modelo,feature,feature_i,phi_r,N=1000):
  optimi=tf.keras.optimizers.Adam()
  err=[]
  for i in list(range(N)):
    grad=perdida_gradiente(modelo,feature,feature_i,phi_r)
    loss=grad[0]
    value=grad[1]
    optimi.apply_gradients(zip(value,modelo.trainable_weights))
    if i%100==0:
      print("Entrenamiento :",i,"perdida :",float(loss))
    err.append(loss)
  return err
inicio = time.time()



L=20

x=np.linspace(0.0,20.0,L)
y=np.linspace(0.0,20.0,L)
t=np.linspace(0.0,20.0,L)

var=list(it.product(t,x,y))

t_col=np.array(var,dtype=np.float32)[:,0][:,None]
x_col=np.array(var,dtype=np.float32)[:,1][:,None]
y_col=np.array(var,dtype=np.float32)[:,2][:,None]

#idx=[np.random.choice(t.shape[0],20, replace=False)]
#t_col=t_col[idx,:][0]


size=t_col.shape[0]

x_i=np.array(np.full((size,),[0.0,]),dtype=np.float32)[:,None]
y_i=np.array(np.full((size,),[0.0,]),dtype=np.float32)[:,None]
x_f=np.array(np.full((size,),[20.0,]),dtype=np.float32)[:,None]
y_f=np.array(np.full((size,),[20.0,]),dtype=np.float32)[:,None]


#pensando en V=E0*y+C

phi_r=np.array([0.0,],dtype=np.float32)[:,None]
phix_r=np.array([0.0,],dtype=np.float32)[:,None]
phiy_r=np.array([5.0,],dtype=np.float32)[:,None]

#ph = tf.placeholder(shape=[None,2], dtype=tf.float32)

inputs=tf.keras.Input(shape=[3,], dtype=tf.float32)
capa1=tf.keras.layers.Dense(units=64,activation=tf.tanh)(inputs)
capa2=tf.keras.layers.Dense(units=64,activation=tf.tanh)(capa1)
capa3=tf.keras.layers.Dense(units=64,activation=tf.tanh)(capa2)
capa4=tf.keras.layers.Dense(units=64,activation=tf.tanh)(capa3)#units = cantidad de naurona
outputs=tf.keras.layers.Dense(units=1,activation=None)(capa4)#dense : neurona que esta conectada con todas las anteriores

model=tf.keras.Model(inputs,outputs)


feature=np.concatenate((x_col,y_col,t_col),axis=1,dtype=np.float32)
t_x=np.array(list(it.product(t,x)),dtype=np.float32)
t_y=np.array(list(it.product(t,y)),dtype=np.float32)
feature_i=np.concatenate((t_col,x_col,y_col,x_i,y_i,x_f,y_f),axis=1,dtype=np.float32)


fin = time.time()
print("Tiempo antes del entrenamiento",fin-inicio,"segundos")

inicio = time.time()
opt=optimizar(model,feature,feature_i,phi_r)
fin = time.time()
print("Tiempo del entrenamiento",fin-inicio,"segundos")
plt.plot(list(range(len(opt))),opt)
plt.show()


dats=model(feature).numpy().reshape((L,L,L))

xx,yy=np.meshgrid(x,y)

fig, axl = plt.subplots(subplot_kw=dict(projection='3d'))
#line, = axl.plot(X,data[0,:],"-k" )

def actualizarL(i):
    axl.clear()
    axl.set_zlim(-5, 5)
    axl.plot_surface(xx, yy, dats[i,:,:], cmap=cm.coolwarm,linewidth=0, antialiased=False)


chu=FuncAnimation(fig,actualizarL)
chu.save('ecuacion de onda_tf3.gif')
#plt.show()
