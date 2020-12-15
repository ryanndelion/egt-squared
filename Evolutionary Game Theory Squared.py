#!/usr/bin/env python
# coding: utf-8

# This notebook serves as a compilation of the various experiments run for the paper 'Evolutionary Game Theory Squared: Evolving Agents in Evolving Games'. Many of these simulations directly corroborate the theorems presented in the paper, and we have also provided several interactive, 3-dimensional simulations within this notebook that should serve to improve the reader's understanding.

# In[46]:


import pandas as pd
import numpy as np
import matplotlib
from matplotlib import pyplot as plt
import matplotlib.animation as animation
from scipy.integrate import odeint, solve_ivp
from scipy.stats import entropy
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from pathlib import Path

from PIL import Image
from tqdm import tqdm
import math
import random
from skimage import measure
from IPython.display import HTML
from pylab import *

get_ipython().run_line_magic('matplotlib', 'inline')


# ## Poincare Recurrence in Time-Evolving RPS

# Main function to simulate 3-strategy RPS with doubly evolving matrices. Uses odeint to integrate the differential equations and returns a dictionary that contains all necessary data.

# In[2]:


P = np.matrix([ [0, -1, 1],
                [1, 0, -1],
                [-1, 1, 0] ])


# In[3]:


def RPSderiv(s,t,P,mu):
    """ Defines the ODE for time evolving RPS. 
    
    Parameters:
    s (array): Array-like of initial conditions [y1, y2, y3, w1, w2, w3]
    t (int): Time to integrate function over
    P (matrix): Standard RPS payoff matrix
    mu (float): Parameter for time evolving system
    
    Returns:
    array: concatenated derivatives of x and w
    """
    x = np.array([[s[0]], [s[1]], [s[2]]])
    w = np.array([[s[3]], [s[4]], [s[5]]])

    A = np.matrix([ [0, (x[1][0]-x[0][0]), (x[2][0]-x[0][0])],
                    [(x[0][0]-x[1][0]), 0, (x[2][0]-x[1][0])],
                    [(x[0][0]-x[2][0]), (x[1][0]-x[2][0]), 0] ])
    B = np.matrix([ [0, (w[0][0]-w[1][0]), (w[0][0]-w[2][0])],
                     [(w[1][0]-w[0][0]), 0, (w[1][0]-w[2][0])],
                     [(w[2][0]-w[0][0]), (w[2][0]-w[1][0]), 0] ])
    Pw = P + mu*B
    dxdt = np.multiply(x, np.matmul(Pw, x)-x.T@Pw@x).flatten().tolist()[0]
    dwdt = np.multiply(w, A@w - w.T@A@w).flatten().tolist()[0]
    
    return np.array(dxdt+dwdt)

def RPSTrajectory(f=RPSderiv, s=[0.1, 0.3, 0.6, 0.1, 0.3, 0.6],
    timestep=0.1, numstep=2000, mu=0.1, P=P) :
    """ Runs ODEint for the RPS system
    
    Parameters:
    s (array): Array-like of initial conditions [x1, x2, x3, w1, w2, w3]
    timestep (float): Timestep of each iteration of the integration
    numstep (int): Number of iterations to be performed
    P (matrix): Standard RPS payoff matrix
    mu (float): Parameter for time evolving system
    
    Returns: 
    dict: Keys are (times, x1, x2, x3, w1, w2, w3)
    The values associated with the keys are time series.
    """
    partuple=(P, mu)        # Converts parameters to a tuple in the right order
    tvals=np.arange(numstep)*timestep
    traj=odeint(f,s,tvals,partuple)
    # Store the results of odeint in a dictionary
    data={}
    data["times"]=tvals
    data["y1"]=traj[:,0]
    data["y2"]=traj[:,1]
    data["y3"]=traj[:,2]
    data["w1"]=traj[:,3]
    data["w2"]=traj[:,4]
    data["w3"]=traj[:,5]
    return data


# Run the function to get data for plotting

# In[4]:


s1=[0.1, 0.2, 0.7, 0.7, 0.2, 0.1]
data = RPSTrajectory(s=s1, f=RPSderiv, numstep=10000)


# ### Plot phase portraits

# In[5]:


import plotly.graph_objects as go
import pandas as pd
import numpy as np

x = data['y1']
y = data['y2']
z = data['y3']

p1=[1, 0, 0, 1]
p2=[0, 1, 0, 0]
p3=[0, 0, 1, 0]

fig = go.Figure([go.Scatter3d(mode='markers', x=x, y=y, z=z, marker=dict(size=2,color=z,colorscale='viridis'),showlegend=False),
                 go.Scatter3d(mode='lines', x=p1, y=p2, z=p3, line=dict(color='black', width=2),showlegend=False)])

fig.update_layout(template="seaborn",
                  title='Phase Portrait for Y player',
                   font=dict(size=15),
                   scene=dict(xaxis_title='Y1',
                              yaxis_title='Y2',
                              zaxis_title='Y3',
                              aspectratio = dict(x=1, y=1, z=0.7),))


# In[6]:


x2 = data['w1']
y2 = data['w2']
z2 = data['w3']

p1=[1, 0, 0, 1]
p2=[0, 1, 0, 0]
p3=[0, 0, 1, 0]

fig1 = go.Figure([go.Scatter3d(mode='markers', x=x2, y=y2, z=z2, 
                               marker=dict(size=2, color=z, colorscale='agsunset',),showlegend=False),
                  go.Scatter3d(mode='lines', x=p1, y=p2, z=p3, line=dict(color='black',width=2),showlegend=False)])

fig1.update_layout(template="seaborn",
                  title='Phase Portrait for W player',
                   font=dict(size=15),
                   scene=dict(xaxis_title='W1',
                              yaxis_title='W2',
                              zaxis_title='W3',
                              aspectratio = dict(x=1, y=1, z=0.7),))


# ## Constant Weighted Sum of KL-Divergence

# The system exhibits a constant of motion, which indicates that volume preservation holds in the system.

# In[7]:


s2=[0.1, 0.2, 0.7, 0.9, 0.05, 0.05]
mu2=0.9
data_kl = RPSTrajectory(s=s2, mu=mu2, f=RPSderiv, numstep=5000)


# In[8]:


data_y = np.array([data_kl[i] for i in ['y1', 'y2', 'y3']]).T
data_w = np.array([data_kl[i].T for i in ['w1', 'w2', 'w3']]).T


# We show that the standard method of determining recurrence fails. That is, if we check the KL divergence between the evolving Nash and strategies, we see that the KL divergence is not constant.

# In[9]:


data_nash=[]
for game in data_w:
    denom = (-3+mu2*(game[0]-game[1])-mu2*(game[0]-game[2])+mu2*(game[1]-game[2]))
    y1 = -(1-mu2*(game[1]-game[2]))/denom
    y2 = -(1+mu2*(game[0]-game[2]))/denom
    y3 = -(1-mu2*(game[0]-game[1]))/denom
    data_nash.append([y1,y2,y3])
data_nash=np.array(data_nash)


# In[10]:


div_y = []
for i in range(len(data_y)):
#     print(i)
    kl_div_y = entropy(data_nash[i], qk=data_y[i])
    div_y.append(kl_div_y)

y_weighted = [y for y in div_y]

div_w = []
for i in range(len(data_w)):
    kl_div_w = entropy(data_nash[i], qk=data_w[i])
    div_w.append(kl_div_w)

w_weighted = [mu2*w for w in div_w]

div_combined = np.add(w_weighted, y_weighted)


# In[11]:



fig = go.Figure([go.Scatter(y=data_nash[:,0][:1000],
                    mode='lines', line=dict(width=1),
                    name='y1 Nash'), 
                 go.Scatter(y=data_nash[:,1][:1000],
                    mode='lines',
                    name='y2 Nash', line=dict(width=1)),
                 go.Scatter(y=data_nash[:,2][:1000],
                    mode='lines',
                    name='y3 Nash', line = dict(width = 1))
                ])

# Edit layout
fig.update_layout(title='Evolution of y-player Nash Equilibrium over time',
                  xaxis_title='Time Steps',
                  yaxis_title='',
                  legend_orientation='h', 
                  legend=dict( y=-0.2),
                  font=dict(size=18))


# In[12]:


fig = go.Figure([go.Scatter(y=data_y[:,0][:1000],
                    mode='lines', line=dict(width=1),
                    name='y1'), 
                 go.Scatter(y=data_y[:,1][:1000],
                    mode='lines',
                    name='y2', line=dict(width=1)),
                 go.Scatter(y=data_y[:,2][:1000],
                    mode='lines',
                    name='y3', line = dict(width = 1))
                ])

# Edit layout
fig.update_layout(title='Evolution of y strategies over time',
                  xaxis_title='Time Steps',
                  yaxis_title='',
                  legend_orientation='h', 
                  legend=dict( y=-0.2),
                  font=dict(size=18))


# In[13]:


fig = go.Figure([go.Scatter(y=div_y[:2000],
                    mode='lines', line=dict(width=1.5))])

# Edit layout
fig.update_layout(title='KL-Divergence between strategy and Nash',
                  xaxis_title='Time Steps',
                  yaxis_title='KL-Divergence',
                  legend_orientation='h', 
                  legend=dict( y=-0.2),
                  font=dict(size=18))


# We verify Lemma 4.1 in the paper by showing that there is a constant of motion so that the orbit remains bounded.

# In[14]:


div_y = []
for i in data_y:
    kl_div_y = entropy([1/3, 1/3, 1/3], qk=i)
    div_y.append(kl_div_y)

y_weighted = [y for y in div_y]

div_w = []
for i in data_w:
    kl_div_w = entropy([1/3, 1/3, 1/3], qk=i)
    div_w.append(kl_div_w)

w_weighted = [mu2*w for w in div_w]

div_combined = np.add(w_weighted, y_weighted)


# In[15]:


fig = go.Figure([go.Scatter(y=y_weighted[:1000],
                    mode='lines', line=dict(width=0.5, color='#4a69bb'),
                    name='Weighted y', fill='tozeroy'), 
                 go.Scatter(y=div_combined[:1000],
                    mode='lines',
                    name='Weighted w', line=dict(width=0.5, color='#6ece58'), fill='tonexty'),
                 go.Scatter(y=div_combined[:1000],
                    mode='lines',
                    name='Sum of Divergences', line = dict(width = 3, color='#440154'), opacity=1)
                ])

# Edit layout
fig.update_layout(title='Constant of Motion for 2-player RPS game',
                  xaxis_title='Time Steps',
                  yaxis_title='KL-Divergence',
                  legend_orientation='h', 
                  legend=dict( y=-0.2),
                  font=dict(size=15))


# ## 5 node zero-sum polymatrix game

# General form (using matrix multiplication) of the time evolving RPS setting. We use this to show that even for complex systems, the volume preservation property holds. 

# In[16]:


def RPSderiv5node(s,t,P,mu,orig=False):
    """ Defines the ODE for time evolving RPS (5 node). 
    
    Parameters:
    s (array): Array-like of initial conditions [y1, y2, y3, w1, w2, w3]
    t (int): Time to integrate function over
    P (matrix): Standard RPS payoff matrix
    mu (float): Parameter for time evolving system
    
    Returns:
    array: concatenated derivatives of x and w
    """
    
    x1 = np.array([[s[0]], [s[1]], [s[2]]])
    x2 = np.array([[s[3]], [s[4]], [s[5]]]) #w1
    x3 = np.array([[s[6]], [s[7]], [s[8]]])
    x4 = np.array([[s[9]], [s[10]], [s[11]]]) #w2
    x5 = np.array([[s[12]], [s[13]], [s[14]]])
    
    mu1, mu2, mu3, mu4 = mu
    

    A1 = np.matrix([ [0, (x1[1][0]-x1[0][0]), (x1[2][0]-x1[0][0])],
                    [(x1[0][0]-x1[1][0]), 0, (x1[2][0]-x1[1][0])],
                    [(x1[0][0]-x1[2][0]), (x1[1][0]-x1[2][0]), 0] ])
#     print(A1)
    A2 = np.matrix([ [0, (x2[1][0]-x2[0][0]), (x2[2][0]-x2[0][0])],
                    [(x2[0][0]-x2[1][0]), 0, (x2[2][0]-x2[1][0])],
                    [(x2[0][0]-x2[2][0]), (x2[1][0]-x2[2][0]), 0] ])
    A3 = np.matrix([ [0, (x3[1][0]-x3[0][0]), (x3[2][0]-x3[0][0])],
                    [(x3[0][0]-x3[1][0]), 0, (x3[2][0]-x3[1][0])],
                    [(x3[0][0]-x3[2][0]), (x3[1][0]-x3[2][0]), 0] ])
    # Note that these two matrices are defined like we defined the B's so essentially
    # as the negation of how the above three are defined, which is why there is some negative 
    # signs in the dynamics
    A4 = np.matrix([ [0, (x4[1][0]-x4[0][0]), (x4[2][0]-x4[0][0])], 
                     [(x4[0][0]-x4[1][0]), 0, (x4[2][0]-x4[1][0])],
                     [(x4[0][0]-x4[2][0]), (x4[1][0]-x4[2][0]), 0] ])
    A5 = np.matrix([ [0, (x5[1][0]-x5[0][0]), (x5[2][0]-x5[0][0])],
                     [(x5[0][0]-x5[1][0]), 0, (x5[2][0]-x5[1][0])],
                     [(x5[0][0]-x5[2][0]), (x5[1][0]-x5[2][0]), 0] ])
    I = np.matrix([[1., 0., 0.], 
                   [0., 1., 0.], 
                   [0., 0., 1.]])
    
    P = np.matrix([[0., -1., 1.],
                   [1., 0., -1.],
                   [-1., 1., 0.] ])
    P1w = P - mu1*A2 
    P2w = P + 1*A2 - mu3*A4 
    P3w = P + 1*A4 
    
    dx1dt = np.multiply(x1, np.matmul(P1w, x1)-x1.T@P1w@x1).flatten().tolist()[0] #x11
    
    dx2dt = np.multiply(x2, (A1-mu2*A3)@x2 - x2.T@(A1-mu2*A3)@x2).flatten().tolist()[0] #x2
    
    dx3dt = np.multiply(x3, np.matmul(P2w, x3)-x3.T@P2w@x3).flatten().tolist()[0] #x3
    
    dx4dt = np.multiply(x4, (A3-mu4*A5)@x4 - x4.T@(A3+mu4*A5)@x4).flatten().tolist()[0] #x4
    
    dx5dt = np.multiply(x5, np.matmul(P3w, x5)-x5.T@P3w@x5).flatten().tolist()[0] #x5

    return np.array(dx1dt+dx2dt+dx3dt +dx4dt+dx5dt)

def RPSTrajectory5node(s, f=RPSderiv5node,
    timestep=0.1, numstep=1000, mu=0.1, P=P) :
    """ Runs ODEint for the RPS system
    
    Parameters:
    s (array): Array-like of initial conditions [x1, x2, x3, w1, w2, w3]
    timestep (float): Timestep of each iteration of the integration
    numstep (int): Number of iterations to be performed
    P (matrix): Standard RPS payoff matrix
    mu (float): Parameter for time evolving system
    
    Returns: 
    dict: Keys are (times, x1, x2, x3, w1, w2, w3)
    The values associated with the keys are time series.
    """
    partuple=(P, mu)        # Converts parameters to a tuple in the right order
    tvals=np.arange(numstep)*timestep
    traj=odeint(f,s,tvals,partuple)

    # Store the results of odeint in a dictionary
    data={}
    data["times"]=tvals
    data["x1"] = traj[:,:3]
    data["x2"] = traj[:,3:6]
    data["x3"] = traj[:,6:9]
    data["x4"] = traj[:,9:12]
    data["x5"] = traj[:,12:15]
    return data


# Get data for a 5-node polymatrix RPS-like game. For this section, $x_i$ is a matrix with 3 columns representing the RPS strategies of agent $i$. 

# In[17]:


s3 = [0.3, 0.4, 0.3, 0.2, 0.1, 0.7, 0.5, 0.3, 0.2, 0.7, 0.2, 0.1, 0.4, 0.2, 0.4]
mu3 = [0.1, 0.5, 0.8, 0.5]
data_5node = RPSTrajectory5node(s=s3, numstep=2000, f=RPSderiv5node, mu=mu3)


# We use the scipy entropy function to compute the KL-divergence of these strategies from the Nash equilibrium.

# In[18]:


div_x1 = []
for i in data_5node['x1']:
    kl_div_x1 = entropy([1/3, 1/3, 1/3], qk=i)
    div_x1.append(kl_div_x1)

x1_weighted = np.array([x for x in div_x1])

div_x2 = []
for i in data_5node['x2']:
    kl_div_x2 = entropy([1/3, 1/3, 1/3], qk=i)
    div_x2.append(kl_div_x2)

x2_weighted = np.array([mu3[0]*x for x in div_x2])

div_x3 = []
for i in data_5node['x3']:
    kl_div_x3 = entropy([1/3, 1/3, 1/3], qk=i)
    div_x3.append(kl_div_x3)
    
x3_weighted = np.array([mu3[0]*mu3[1]*x for x in div_x3])
                    
div_x4 = []
for i in data_5node['x4']:
    kl_div_x4 = entropy([1/3, 1/3, 1/3], qk=i)
    div_x4.append(kl_div_x4)

x4_weighted = np.array([mu3[0]*mu3[1]*mu3[2]*x for x in div_x4])

div_x5 = []
for i in data_5node['x5']:
    kl_div_x5 = entropy([1/3, 1/3, 1/3], qk=i)
    div_x5.append(kl_div_x5)

x5_weighted = np.array([mu3[0]*mu3[1]*mu3[2]*mu3[3]*x for x in div_x5])

div_combined = x1_weighted+x2_weighted+x3_weighted+x4_weighted+x5_weighted


# In[19]:


import plotly.graph_objects as go
fig = go.Figure([go.Scatter(y=x1_weighted,
                    mode='lines', line=dict(width=0.5, color='#fde725'),
                    name='S1', fill='tozeroy'), 
                 go.Scatter(y=x1_weighted+x2_weighted, mode='lines',
                    name='S2', line=dict(width=0.5, color='#6ece58'), fill='tonexty'),
                go.Scatter(y=x1_weighted+x2_weighted+x3_weighted, mode='lines',
                    name='S3', line=dict(width=0.5, color='#4a69bb'), fill='tonexty'),
                 go.Scatter(y=x1_weighted+x2_weighted+x3_weighted+x4_weighted, mode='lines',
                    name='S4', line=dict(width=0.5, color='#ff7315'), fill='tonexty'),
                 go.Scatter(y=x1_weighted+x2_weighted+x3_weighted+x4_weighted+x5_weighted, mode='lines',
                    name='S5', line=dict(width=0.5, color='#fe346e'),opacity=1, fill='tonexty'),
                 go.Scatter(y=div_combined,
                    mode='lines',
                    name='Sum of Divergences', line = dict(width = 3, color='#440154'),opacity=1)
                ])

# Edit layout
fig.update_layout(title='KL-Divergence for 5-node polymatrix game',
                  xaxis_title='Time Steps',
                  yaxis_title='KL-Divergence', 
                  legend_orientation='h', 
                  legend=dict( y=-0.2),
                  font=dict(size=15))


# In[20]:


div_x1 = []
for i in data_5node['x1']:
    kl_div_x1 = -entropy([1/3, 1/3, 1/3], qk=i)+entropy([1/3,1/3,1/3])
    div_x1.append(kl_div_x1)

x1_weighted = np.array([1*x for x in div_x1])

div_x2 = []
for i in data_5node['x2']:
    kl_div_x2 = -(entropy([1/3, 1/3, 1/3], qk=i)-entropy([1/3,1/3,1/3]))
    div_x2.append(kl_div_x2)

x2_weighted = np.array([mu3[0]*x for x in div_x2])

div_x3 = []
for i in data_5node['x3']:
    kl_div_x3 =-(entropy([1/3, 1/3, 1/3], qk=i)-entropy([1/3,1/3,1/3]))
    div_x3.append(kl_div_x3)
    
x3_weighted = np.array([mu3[1]*mu3[0]*x for x in div_x3])
                    
div_x4 = []
for i in data_5node['x4']:
    kl_div_x4 = -(entropy([1/3, 1/3, 1/3], qk=i)-entropy([1/3,1/3,1/3]))
    div_x4.append(kl_div_x4)

x4_weighted = np.array([mu3[0]*mu3[1]*mu3[2]*x for x in div_x4])

div_x5 = []
for i in data_5node['x5']:
    kl_div_x5 = -(entropy([1/3, 1/3, 1/3], qk=i)-entropy([1/3,1/3,1/3]))
    div_x5.append(kl_div_x5)

x5_weighted = np.array([mu3[0]*mu3[1]*mu3[2]*mu3[3]*x for x in div_x5])

div_combined = x1_weighted+x2_weighted+x3_weighted+x4_weighted+x5_weighted


# In[21]:


fig = go.Figure([go.Scatter(y=x1_weighted,
                    mode='lines', line=dict(width=0.5, color='#fde725'),
                    name='S1', fill='tozeroy'), 
                 go.Scatter(y=x1_weighted+x2_weighted, mode='lines',
                    name='S2', line=dict(width=0.5, color='#6ece58'), fill='tonexty'),
                go.Scatter(y=x1_weighted+x2_weighted+x3_weighted, mode='lines',
                    name='S3', line=dict(width=0.5, color='#4a69bb'), fill='tonexty'),
                 go.Scatter(y=x1_weighted+x2_weighted+x3_weighted+x4_weighted, mode='lines',
                    name='S4', line=dict(width=0.5, color='#ff7315'), fill='tonexty'),
                 go.Scatter(y=x1_weighted+x2_weighted+x3_weighted+x4_weighted+x5_weighted, mode='lines',
                    name='S5', line=dict(width=0.5, color='#fe346e'),opacity=1, fill='tonexty'),
                 go.Scatter(y=div_combined,
                    mode='lines',
                    name='Sum of Divergences', line = dict(width = 3, color='#440154'),opacity=1)
                ])

# Edit layout
fig.update_layout(title='Constant of Motion for 5-node polymatrix game',
                  xaxis_title='Time Steps',
                  yaxis_title='$KL(x^* \| x) + \mathcal{I}(x^*)$', 
                  legend_orientation='h', 
                  legend=dict( y=-0.2),
                  font=dict(size=15))


# ## Poincare Sections

# We created several Poincare sections, which show the intersection points of the RPS trajectories with a pre-defined hyperplane. Using these Poincare sections, we show the quasi-periodicity of these time-evolving systems. In addition, we can see that the frequency of intersection depends on the initial conditions of the simulation.

# #### 2D Poincare Sections

# We first used the hyperplane $y_2-y_1+w_2-w_1=0$ and obtained a 2-dimensional plot. 

# In[22]:


mu_2 = 0.8
step = .001
num = 4000000
psectiondata = []
for i in range(1, 10+1):
    init = np.array([0.5, 0.01*i, 0.5-0.01*i, 0.5, 0.25, 0.25])
    t_span = (0, num*step)
    t_eval = np.arange(num)*step
  
    partuple=(P, mu_2)        # Converts parameters to a tuple in the right order
    traj=odeint(RPSderiv,init,t_eval,partuple)
    
    stat = traj[:, 1]-traj[:, 0] + traj[:, 4] - traj[:, 3]
    idx = np.where((stat[1:]*stat[:-1]<0)&(stat[1:]<0))[0] 
    psectiondata.append((traj[idx, 0], traj[idx, 4]))


# In[23]:


plt.figure(figsize=(15, 10))
plt.style.use('seaborn')
for data_point in psectiondata:
    plt.scatter(data_point[0], data_point[1], s=15);
plt.xlabel(r'$y_1$', fontsize=22)
plt.ylabel(r'$w_2$', fontsize=22)
plt.tick_params(axis='x', labelsize=22)
plt.tick_params(axis='y', labelsize=22)
plt.title(r'Poincare Section for hyperplane $y_2-y_1+w_2-w_1=0$', fontsize=22);


# #### 3D Poincare Sections

# In order to better understand the differences between various initial conditions, we ran simulations for the Poincare section intersecting hyperplane $y_2+y_1+w_2+w_1=4/3$, which occurs at the Nash equilibrium. We observe that as the system initiates closer to the mixed Nash, the frequency of intersection with the hyperplane increases.

# In[24]:


def GetRPSPoincare(data):
    """ Generates a Poincare section:
    plots y1 vs w2 when y2+y1+w2+w1=4/3 and w1>1/3.
    
    Parameters:
    data (dict): Dictionary containing time series data from RPS simulations
    
    Returns:
    signchangeloc: array of locations where a hyperplane passes through 0
    
    """
    #unpack data
    #NOTE: here we used hyperplane y1+y2+w1+w2-4/3=0
    tester = (data["y2"]+data["y1"])+(data["w2"]+data["w1"])-4/3
    x1=data["y1"]
    x2=data["y2"]
    x3=data["y3"]
    w1=data["w1"]
    w2=data["w2"]
    w3=data["w3"]
    t=data["times"]
    
    #get indices where the tester is zero and the trajectories pass through the hyperplane
    xsignprod = (tester[:-1]*tester[1:]<0)&(tester[:-1]>0)#&(w1[1:] > 1/3)
    signchangeloc=np.where(xsignprod)[0]
    data['signchangeloc'] = signchangeloc
    data['y1dat'] = x1[signchangeloc]
    data['y2dat'] = x2[signchangeloc]
    data['y3dat'] = x3[signchangeloc]
    data['w1dat'] = w1[signchangeloc]
    data['w2dat'] = w2[signchangeloc]
    data['w3dat'] = w3[signchangeloc]
    print('Poincare section data generated')

    return data

def lininterp(xvals,yvals):
    """ Take (x0,y0) and (x1,y1) to be points defining a line
    finds the value of y when x=0
    
    Parameters:
    xvals, yvals (tuples) : x and y values to interpolate between
    """
    return (xvals[1]*yvals[0]-xvals[0]*yvals[1])/(xvals[1]-xvals[0])


# In[25]:


def GetIteratedPoincare(numiters=25, numstep=10000, timestep=0.1,mu=0.1, agents_to_plot=['x1', 'x2', 'w1']):
    '''Runs a defined number of iterations of the GetRPSPoincare function, and adds relevant data to the data dictionary.
    
    Parameters:
    numiters (int): Number of iterations to run the simulation. Note that due to the defined initial conditions, the maximum number of iterations is 50
    numstep (int): Number of steps to run the RPS simulation for each iteration
    timestep (float): Timestep of each iteration of the integration
    agents_to_plot (array-like): 3 of the agents in the form [X, Y, Z]. These 3 agents will be saved in the output, which can then be plotted in a 3D graph
    
    Returns:
    data (dict): Updated dictionary that includes Poincare section data for the agents defined
    
    '''
    if numiters>50:
        print('Maximum iterations is 50!')
        return
    
    data=dict()
    k = np.linspace(1, numiters, num=numiters)
    poincare_data_x = []
    poincare_data_y = []
    poincare_data_z = []
    for i in k:
        print("Iteration: {}".format(int(i)))
        init = [1/3, 0.03*i, 2/3-0.03*i, 1/3, 1/3, 1/3]
        data_i = RPSTrajectory(s=init, numstep=numstep, timestep=timestep, mu=mu)
        poincare = GetRPSPoincare(data_i)
        poincare_data_x.append(poincare[agents_to_plot[0]+'dat'])
        poincare_data_y.append(poincare[agents_to_plot[1]+'dat'])
        poincare_data_z.append(poincare[agents_to_plot[2]+'dat'])
        plt.scatter(poincare[agents_to_plot[0]+'dat'], poincare[agents_to_plot[1]+'dat'], s=1)

    data['poincare_data_x'] = poincare_data_x
    data['poincare_data_y'] = poincare_data_y
    data['poincare_data_z'] = poincare_data_z
    data['agents_to_plot'] = agents_to_plot
    return data


# In[26]:


poincare_data = GetIteratedPoincare(numiters=10, numstep=100000, mu=0.8, agents_to_plot=['y1', 'w2', 'w3'])


# In[27]:


import itertools
#combining all iterations of each agent in one big list (we don't really need to care which iteration gives the relevant data)
poincare_data_x_joined = list(itertools.chain.from_iterable(poincare_data['poincare_data_x']))
poincare_data_y_joined = list(itertools.chain.from_iterable(poincare_data['poincare_data_y']))
poincare_data_z_joined = list(itertools.chain.from_iterable(poincare_data['poincare_data_z']))


# In[28]:


z1 = np.arange(0,10, 1)  


traces =  [go.Scatter3d(x=poincare_data['poincare_data_x'][k], y=poincare_data['poincare_data_y'][k], z=poincare_data['poincare_data_z'][k], mode='markers',
                              marker=dict(size=1.5),)
           for k in z1]

layout = go.Layout(width=900,
                   height=700,
                   autosize=False,
                   showlegend=False, 
                   scene= dict(
                       xaxis=dict(title= poincare_data['agents_to_plot'][0]),
                       yaxis=dict(title= poincare_data['agents_to_plot'][1]),
                       zaxis=dict(title= poincare_data['agents_to_plot'][2]),
                       aspectratio=dict(x=1, y=1, z=0.5))
    
                  )

fig = go.Figure(data= traces, layout=layout)
fig.show()


# ## Time Average Convergence

# Another key set of theorems is that given the existence of a unique, fully-mixed Nash equilibrium, the time-average of the replicator dynamics converge to the equilibrium strategies for each player, and the time-average utility converges to the utility of the equilibrium. We confirm this statement experimentally by running the RPS game for a set of random initial conditions and checking that these time averages all converge.

# In[29]:


def run_rps(vec, time, f, P, mu):

    t = np.arange(time)*.01
    results = odeint(f, vec, t, args=(P, mu))
    
    return results


# ### Time averages for 2 player case

# Run the 2-player RPS game for a set of initial conditions, and compute the expected utilities for each player.

# In[30]:


x_regret = []
w_regret = []

x_regret_loop = []
w_regret_loop = []


num_init = 10
#initial_conditions = []
entropies = []

N=10
x1_inits = np.linspace(0.1,0.7, N)
x = np.asarray([[x1, 0.75-x1, 0.25] for x1 in x1_inits])
w=np.random.rand(10,3)
initial_conditions=np.hstack((x,w)) #.append([x,w])
wutilities=[]
xutilities=[]
xvalstime=[]
wvals1=[];  wvals2=[]; wvals3=[]
xvals1=[];  xvals2=[]; xvals3=[]
for x0 in initial_conditions:
    x=x0[0:3]
    w=x0[3:6]

    x = x/x.sum()
    w = w/w.sum()
    vec = np.concatenate([x, w])
    
    print('Initial Conditions: ', vec)

    time = 20000
    mu = 0.8
    rpsdata = run_rps(vec, time=time, f=RPSderiv, P=P, mu=mu)
    A = mu*np.eye(3)
    B = -np.eye(3)

    utility = []

    x1s=[]; x2s=[]; x3s=[] # local storage
    for i in range(len(rpsdata)):
        regret = ((np.array([1/3,1/3,1/3])@A@np.array([1/3,1/3,1/3])+np.array([1/3,1/3,1/3])@P@np.array([1/3,1/3,1/3]) 
                   - rpsdata[i, :3]@A@rpsdata[i, 3:]  - rpsdata[i, :3]@P@rpsdata[i, :3]).item())
        xval = (rpsdata[i, :3])
        utility.append(regret)
        x1s.append(xval[0])
        x2s.append(xval[1])
        x3s.append(xval[2])
    xutilities.append(utility)
    xvals1.append(x1s)
    xvals2.append(x2s)
    xvals3.append(x3s)


    x_regret_loop.append(utility)

    utility = []

    w1s=[]; w2s=[]; w3s=[]
    for i in range(len(rpsdata)):
        regret = (np.array([1/3,1/3,1/3])@B@np.array([1/3,1/3,1/3])- rpsdata[i, :3]@B@rpsdata[i, 3:]).item()
        utility.append(regret)
        wval = (rpsdata[i, 3:])

        w1s.append(wval[0])
        w2s.append(wval[1])
        w3s.append(wval[2])

    wvals1.append(w1s)
    wvals2.append(w2s)
    wvals3.append(w3s)

    w_regret_loop.append(utility)
    wutilities.append(utility)

cumsums=[]
for utilval in xutilities:
    cumsums.append(np.cumsum(utilval)/np.arange(1, len(utilval)+1))
cumsumsw=[]
for utilval in wutilities:
    cumsumsw.append(np.cumsum(utilval)/np.arange(1, len(utilval)+1))
    
cxvals1=[]
for xvalit in xvals1:
    cxvals1.append(np.cumsum(xvalit)/np.arange(1, len(xvalit)+1))
    
cxvals2=[]
for xvalit in xvals2:
    cxvals2.append(np.cumsum(xvalit)/np.arange(1, len(xvalit)+1))
    
cxvals3=[]
for xvalit in xvals3:
    cxvals3.append(np.cumsum(xvalit)/np.arange(1, len(xvalit)+1))
    
cwvals1=[]
for utilval in wvals1:
    cwvals1.append(np.cumsum(utilval)/np.arange(1, len(utilval)+1))
    
cwvals2=[]
for utilval in wvals2:
    cwvals2.append(np.cumsum(utilval)/np.arange(1, len(utilval)+1))
    
cwvals3=[]
for utilval in wvals3:
    cwvals3.append(np.cumsum(utilval)/np.arange(1, len(utilval)+1))


# In[31]:


uxmeans=np.mean(np.asarray(cumsums),axis=0)
uxvars=np.std(np.asarray(cumsums),axis=0)

uwmeans=np.mean(np.asarray(cumsumsw),axis=0)
uwvars=np.std(np.asarray(cumsumsw),axis=0)

plt.figure(figsize=(10,8));
plt.plot(uxmeans, color='tab:red', linewidth=3, label=r'$\hat{u}_y(t)-u_y^\ast$');
plt.plot(uxmeans+uxvars,color='tab:blue', label=r'$\pm 1$std');
plt.plot(uxmeans-uxvars,color='tab:blue');

plt.fill_between(np.arange(0,20000),uxmeans-uxvars,uxmeans+uxvars,color='tab:blue', alpha=0.5)

plt.tick_params(labelsize=22)
plt.legend(fontsize=22)
plt.title(r'Time Average Utility ($y$-player)', fontsize=22);


# Plot the time average strategies for the $y$ player, as well as the corresponding variances.

# In[32]:


# import seaborn as sns
# sns.set_style("darkgrid")
xmeans1=np.mean(np.asarray(cxvals1),axis=0)
xvars1=np.std(np.asarray(cxvals1),axis=0)
xmeans2=np.mean(np.asarray(cxvals2),axis=0)
xvars2=np.std(np.asarray(cxvals2),axis=0)
xmeans3=np.mean(np.asarray(cxvals3),axis=0)
xvars3=np.std(np.asarray(cxvals3),axis=0)
x1col='tomato'
plt.figure(figsize=(10,8))
#plt.plot(xmeans1, color='tab:red', linewidth=3, label=r'$\hat{x}_1(t)-x_1^\ast$')
plt.plot(xmeans1+xvars1,color=x1col) # label=r'$\pm 1$std')
plt.plot(xmeans1-xvars1,color=x1col)
plt.fill_between(np.arange(0,20000),xmeans1-xvars1,xmeans1+xvars1,color=x1col, alpha=0.5)

x2col='deepskyblue'
#plt.plot(xmeans2, color='tab:blue', linewidth=3, label=r'$\hat{x}_2(t)-x_2^\ast$')
plt.plot(xmeans2+xvars2,color=x2col)
plt.plot(xmeans2-xvars2,color=x2col)
plt.fill_between(np.arange(0,20000),xmeans2-xvars2,xmeans2+xvars2,color=x2col, alpha=0.5)

x3col='khaki'

plt.plot(xmeans3+xvars3,color=x3col)
plt.plot(xmeans3-xvars2,color=x3col)
plt.fill_between(np.arange(0,20000),xmeans3-xvars3,xmeans3+xvars3,color=x3col, alpha=0.5)

plt.plot(xmeans1, color='tab:red', linewidth=3, label=r'$\hat{y}_1(t)-y_1^\ast$')
plt.plot(xmeans2, color='tab:blue', linewidth=3, label=r'$\hat{y}_2(t)-y_2^\ast$')
plt.plot(xmeans3, color='gold', linewidth=3, label=r'$\hat{y}_3(t)-y_3^\ast$')
#plt.plot(wmeans, color='tab:orange', linewidth=3, label=r'$\hat{u}_w(t)-u_w^\ast$')
#plt.plot(wmeans+wvars,color='yellow', label=r'$\pm 1$std')
#plt.plot(wmeans-wvars,color='yellow')

#plt.fill_between(np.arange(0,20000),wmeans-wvars,wmeans+wvars,color='yellow', alpha=0.5)
plt.tick_params(labelsize=22)
plt.legend(fontsize=22)
# plt.show()
plt.title(r'Time Average Actions ($y$-player)', fontsize=22);


# Plot the time average strategies for the $w$ player, as well as the corresponding variances.

# In[33]:



wmeans1=np.mean(np.asarray(cwvals1),axis=0)
wvars1=np.std(np.asarray(cwvals1),axis=0)
wmeans2=np.mean(np.asarray(cwvals2),axis=0)
wvars2=np.std(np.asarray(cwvals2),axis=0)
wmeans3=np.mean(np.asarray(cwvals3),axis=0)
wvars3=np.std(np.asarray(cwvals3),axis=0)
w1col='tomato'
plt.figure(figsize=(10,8));
#plt.plot(xmeans1, color='tab:red', linewidth=3, label=r'$\hat{x}_1(t)-x_1^\ast$')
plt.plot(wmeans1+wvars1,color=w1col); # label=r'$\pm 1$std')
plt.plot(wmeans1-wvars1,color=w1col);
plt.fill_between(np.arange(0,20000),wmeans1-wvars1,wmeans1+wvars1,color=w1col, alpha=0.5)

w2col='deepskyblue'
#plt.plot(xmeans2, color='tab:blue', linewidth=3, label=r'$\hat{x}_2(t)-x_2^\ast$')
plt.plot(wmeans2+wvars2,color=w2col);
plt.plot(wmeans2-wvars2,color=w2col);
plt.fill_between(np.arange(0,20000),wmeans2-wvars2,wmeans2+wvars2,color=w2col, alpha=0.5)

w3col='khaki'

plt.plot(wmeans3+wvars3,color=w3col);
plt.plot(wmeans3-wvars2,color=w3col);
plt.fill_between(np.arange(0,20000),wmeans3-wvars3,wmeans3+wvars3,color=w3col, alpha=0.5)

plt.plot(wmeans1, color='tab:red', linewidth=3, label=r'$\hat{w}_1(t)-w_1^\ast$');
plt.plot(wmeans2, color='tab:blue', linewidth=3, label=r'$\hat{w}_2(t)-w_2^\ast$');
plt.plot(wmeans3, color='gold', linewidth=3, label=r'$\hat{w}_3(t)-w_3^\ast$');
#plt.plot(wmeans, color='tab:orange', linewidth=3, label=r'$\hat{u}_w(t)-u_w^\ast$')
#plt.plot(wmeans+wvars,color='yellow', label=r'$\pm 1$std')
#plt.plot(wmeans-wvars,color='yellow')

#plt.fill_between(np.arange(0,20000),wmeans-wvars,wmeans+wvars,color='yellow', alpha=0.5)
plt.tick_params(labelsize=22)
plt.legend(fontsize=22)
plt.title(r'Time Average Actions ($w$-player)', fontsize=22);


# ### Time Averages for 5 player case

# In[34]:


x1_regret = []
x2_regret = []
x3_regret = []
x4_regret = []
x5_regret = []


x1_regret_loop = []
x2_regret_loop = []
x3_regret_loop = []
x4_regret_loop = []
x5_regret_loop = []


num_init = 10
#initial_conditions = []
entropies = []

N=10
x3_inits = np.linspace(0.1,0.7, N)
x3 = np.asarray([[x, 0.75-x, 0.25] for x in x3_inits])
np.random.seed(4)
x1=np.random.rand(10,3)
np.random.seed(10)
x2=np.random.rand(10,3)
np.random.seed(203)
x4=np.random.rand(10,3)
np.random.seed(78)
x5=np.random.rand(10,3)

initial_conditions=np.hstack((np.hstack((np.hstack((np.hstack((x1,x2)),x3)),x4)),x5)) #.append([x,w])

x1utilities=[]
x2utilities=[]
x3utilities=[]
x4utilities=[]
x5utilities=[]

x1valstime=[]
x2valstime=[]
x3valstime=[]
x4valstime=[]
x5valstime=[]

wvals1=[];  wvals2=[]; wvals3=[]
x1vals1=[];  x1vals2=[]; x1vals3=[]
x2vals1=[];  x2vals2=[]; x2vals3=[]
x3vals1=[];  x3vals2=[]; x3vals3=[]
x4vals1=[];  x4vals2=[]; x4vals3=[]
x5vals1=[];  x5vals2=[]; x5vals3=[]
for x0 in initial_conditions:
    x1=x0[0:3]/x0[0:3].sum()
    x2=x0[3:6]/x0[3:6].sum()
    x3=x0[6:9]/x0[6:9].sum()
    x4=x0[9:12]/x0[9:12].sum()
    x5=x0[12:15]/x0[12:15].sum()


    vec = np.concatenate([x1, x2,x3,x4,x5])
    
    print('Initial Conditions: ', vec)

    time = 20000
    mu = 0.8
    rpsdata = run_rps(vec, time=time, f=RPSderiv5node, P=P, mu=mu3)
    A = mu3[2]*np.eye(3)
    B = -np.eye(3)

    utility = []

    x1s=[]; x2s=[]; x3s=[] # local storage
    for i in range(len(rpsdata)):
        regret = ((np.array([1/3,1/3,1/3])@A@np.array([1/3,1/3,1/3])+np.array([1/3,1/3,1/3])@P@np.array([1/3,1/3,1/3]) 
                       +np.array([1/3,1/3,1/3])@B@np.array([1/3,1/3,1/3])
                       - rpsdata[i, 6:9]@A@rpsdata[i, 9:12]  - rpsdata[i, 6:9]@P@rpsdata[i, 6:9]-rpsdata[i, 6:9]@B@rpsdata[i, 3:6]).item())
        xval = (rpsdata[i, :3])
        utility.append(regret)
        x1s.append(xval[0])
        x2s.append(xval[1])
        x3s.append(xval[2])
    x3utilities.append(utility)
    x3vals1.append(x1s)
    x3vals2.append(x2s)
    x3vals3.append(x3s)
    #x_regret_loop.append(utility)
    if 0:
        x1s=[]; x2s=[]; x3s=[] # local storage
        for i in range(len(rpsdata)):
            #regret = ((np.array([1/3,1/3,1/3])@A@np.array([1/3,1/3,1/3])+np.array([1/3,1/3,1/3])@P@np.array([1/3,1/3,1/3]) 
            #           - rpsdata[i, :3]@A@rpsdata[i, 3:]  - rpsdata[i, :3]@P@rpsdata[i, :3]).item())
            xval = (rpsdata[i, :3])
            #utility.append(regret)
            x1s.append(xval[0])
            x2s.append(xval[1])
            x3s.append(xval[2])
        #xutilities.append(utility)
        x1vals1.append(x1s)
        x1vals2.append(x2s)
        x1vals3.append(x3s)

        x1s=[]; x2s=[]; x3s=[] # local storage
        for i in range(len(rpsdata)):
            #regret = ((np.array([1/3,1/3,1/3])@A@np.array([1/3,1/3,1/3])+np.array([1/3,1/3,1/3])@P@np.array([1/3,1/3,1/3]) 
            #           - rpsdata[i, :3]@A@rpsdata[i, 3:]  - rpsdata[i, :3]@P@rpsdata[i, :3]).item())
            xval = (rpsdata[i, :3])
            #utility.append(regret)
            x1s.append(xval[0])
            x2s.append(xval[1])
            x3s.append(xval[2])
        #xutilities.append(utility)
        x2vals1.append(x1s)
        x2vals2.append(x2s)
        x2vals3.append(x3s)

        utility=[]
        x1s=[]; x2s=[]; x3s=[] # local storage
        for i in range(len(rpsdata)):
            
            xval = (rpsdata[i, :3])

            x1s.append(xval[0])
            x2s.append(xval[1])
            x3s.append(xval[2])
        x4vals1.append(x1s)
        x4vals2.append(x2s)
        x4vals3.append(x3s)

if 0:
    cumsums=[]
    for utilval in xutilities:
        cumsums.append(np.cumsum(utilval)/np.arange(1, len(utilval)+1))
    cumsumsw=[]
    for utilval in wutilities:
        cumsumsw.append(np.cumsum(utilval)/np.arange(1, len(utilval)+1))

    cxvals1=[]
    for xvalit in xvals1:
        cxvals1.append(np.cumsum(xvalit)/np.arange(1, len(xvalit)+1))

    cxvals2=[]
    for xvalit in xvals2:
        cxvals2.append(np.cumsum(xvalit)/np.arange(1, len(xvalit)+1))
    
cx3vals1=[]
for xvalit in x3vals1:
    cx3vals1.append(np.cumsum(xvalit)/np.arange(1, len(xvalit)+1))
    
cx3vals2=[]
for xvalit in x3vals2:
    cx3vals2.append(np.cumsum(xvalit)/np.arange(1, len(xvalit)+1))
    
cx3vals3=[]
for xvalit in x3vals3:
    cx3vals3.append(np.cumsum(xvalit)/np.arange(1, len(xvalit)+1))
    
cumsums=[]
for utilval in x3utilities:
    cumsums.append(np.cumsum(utilval)/np.arange(1, len(utilval)+1))


# In[35]:


xmeans1=np.mean(np.asarray(cx3vals1),axis=0)
xvars1=np.std(np.asarray(cx3vals1),axis=0)
xmeans2=np.mean(np.asarray(cx3vals2),axis=0)
xvars2=np.std(np.asarray(cx3vals2),axis=0)
xmeans3=np.mean(np.asarray(cx3vals3),axis=0)
xvars3=np.std(np.asarray(cx3vals3),axis=0)
x1col='tomato'
plt.figure(figsize=(10,8))
plt.plot(xmeans1+xvars1,color=x1col)
plt.plot(xmeans1-xvars1,color=x1col)
plt.fill_between(np.arange(0,20000),xmeans1-xvars1,xmeans1+xvars1,color=x1col, alpha=0.5)

x2col='deepskyblue'
plt.plot(xmeans2+xvars2,color=x2col)
plt.plot(xmeans2-xvars2,color=x2col)
plt.fill_between(np.arange(0,20000),xmeans2-xvars2,xmeans2+xvars2,color=x2col, alpha=0.5)

x3col='khaki'

plt.plot(xmeans3+xvars3,color=x3col)
plt.plot(xmeans3-xvars2,color=x3col)
plt.fill_between(np.arange(0,20000),xmeans3-xvars3,xmeans3+xvars3,color=x3col, alpha=0.5)

plt.plot(xmeans1, color='tab:red', linewidth=3, label=r'$\hat{x}_{31}(t)-x_{31}^\ast$')
plt.plot(xmeans2, color='tab:blue', linewidth=3, label=r'$\hat{x}_{32}(t)-x_{32}^\ast$')
plt.plot(xmeans3, color='gold', linewidth=3, label=r'$\hat{x}_{33}(t)-x_{33}^\ast$')


plt.tick_params(labelsize=22)
plt.legend(fontsize=22)

plt.title(r'Time Average Actions ($x_3$-player)', fontsize=22);


# In[36]:


uxmeans=np.mean(np.asarray(cumsums),axis=0)
uxvars=np.std(np.asarray(cumsums),axis=0)


plt.figure(figsize=(10,8));
plt.plot(uxmeans, color='tab:red', linewidth=3, label=r'$\hat{u}_3(t)-u_3^\ast$');
plt.plot(uxmeans+uxvars,color='tab:blue', label=r'$\pm 1$std');
plt.plot(uxmeans-uxvars,color='tab:blue');

plt.fill_between(np.arange(0,20000),uxmeans-uxvars,uxmeans+uxvars,color='tab:blue', alpha=0.5)

plt.tick_params(labelsize=22)
plt.legend(fontsize=22)
plt.title(r'Time Average Utility ($x_3$-player)', fontsize=22);


# ## 3D Embedding

# Finally, we also experimented with methods to present the high-dimensional data in an intuitive manner. For this purpose, we transformed the 6-dimensional dataset into 4 dimensions, and used color as a way of expressing the 4-th dimension. This embedding shows how the trajectories change for different initial conditions with a series of gif animations, which are shown below.

# In[37]:


s3=[0.3, 0.3, 0.4, 0.2, 0.1, 0.7]
data2 = RPSTrajectory(s=s3, f=RPSderiv, numstep=10000)


# To obtain bounded embeddings of the 6-dimensional data, we used the following transformation:
# 
# $$y_1' = \frac{\sqrt{2}}{2} y_3 - \frac{\sqrt{2}}{2} y_2, \qquad y_2' =  - \frac{1}{\sqrt{6}} y_3 - \frac{1}{\sqrt{6}} y_2 + \frac{\sqrt{2}}{\sqrt{3}} y_1$$
# $$w_1'= A*\left(\frac{\sqrt{2}}{2} w_3 - \frac{\sqrt{2}}{2} w_2\right), \qquad w_2' = A*\left(- \frac{1}{\sqrt{6}} w_3 - \frac{1}{\sqrt{6}} w_2 + \frac{\sqrt{2}}{\sqrt{3}} w_1\right)$$

# Function to create folder used for saving animation frames

# In[38]:


import os
import imageio
import copy

def createFolder(directory):
    try:
        if not os.path.exists(directory):
            os.makedirs(directory)
    except OSError:
        print ('Error: Creating directory. ' +  directory)


# #### Plotting transformed data

# Function to plot the required elements in the animation. These include colorbar for the 4th dimension of data and simplices representing the initial conditions of the current simulation.

# In[39]:


def plot_save_fig_transform(init, f, name, inner_folder, timestep=0.1, numstep=1000, mu=0.1, A=5, show=False, save=True):
    data = RPSTrajectory(s=init, f=RPSderiv, numstep=numstep, timestep=timestep, mu=mu) 
    init_rounded = [round(x, 3) for x in init]
    X1 = A*((np.sqrt(2)/2)*data['w3'] - (np.sqrt(2)/2)*data['w2'])
    Y1 = A*( -(1/np.sqrt(6))*data['w3'] - (1/np.sqrt(6))*data['w2'] + (np.sqrt(2)/np.sqrt(3))*data['w1'])
    Z1 = np.zeros(len(data['w3']))
    X2 = (np.sqrt(2)/2)*data['y3'] - (np.sqrt(2)/2)*data['y2']
    Y2 = np.zeros(len(data['y3'])) 
    Z2 = -(1/np.sqrt(6))*data['y3'] - (1/np.sqrt(6))*data['y2'] + (np.sqrt(2)/np.sqrt(3))*data['y1']
    
    fig = make_subplots(rows=2, cols=2, column_widths=[0.5, 0.5], row_heights=[0.2, 0.8],
        specs=[[{"type": "ternary"}, {"type": "ternary"}],
               [{"type": "scene", "colspan":2}, None]])
    fig.add_trace(go.Scatterternary(name='Initial Y', a=[init[0]], b=[init[1]], c=[init[2]], mode='markers', 
                                     marker={'symbol': 100,'color': 'green','size': 10},),row=1, col=1),
    fig.add_trace(go.Scatterternary(name='Initial W', a=[init[3]], b=[init[4]], c=[init[5]], mode='markers', 
                                     marker={'symbol': 100,'color': 'red','size': 10},),row=1, col=2),
    fig.add_trace(go.Scatter3d(x=X1, y=Y1, z=X2, mode='markers',showlegend=False,
                                  marker=dict(size=2,cmax=0.5, cmin=-0.5, color=Z2, colorscale='rainbow',colorbar=dict(
                                      title="Transformed Y2 value", thickness=20, len=0.6, y=0.35)),
                                  line=dict(color='black', width=4)),row=2, col=1)

    fig.update_ternaries({
        'aaxis':{'title': 'Y1', 'min': 0.01, 'linewidth':2, 'ticks':'outside' },
        'baxis':{'title': 'Y2', 'min': 0.01, 'linewidth':2, 'ticks':'outside' },
        'caxis':{'title': 'Y3', 'min': 0.01, 'linewidth':2, 'ticks':'outside' }}, col=1)
    
    fig.update_ternaries({
        'aaxis':{'title': 'W1', 'min': 0.01, 'linewidth':2, 'ticks':'outside' },
        'baxis':{'title': 'W2', 'min': 0.01, 'linewidth':2, 'ticks':'outside' },
        'caxis':{'title': 'W3', 'min': 0.01, 'linewidth':2, 'ticks':'outside' }}, col=2)

    fig.update_layout(template="seaborn",
                      title='Inits:'+str(init_rounded),
                       width=900,
                       height=900,
                       autosize=False,
                       showlegend=True,
                       font=dict(size=15),
                       scene=dict(xaxis_title='Transformed W1',
                                  yaxis_title='Transformed W2',
                                  zaxis_title='Transformed Y1',
                                  xaxis = dict(range=[-3.5,3.5]),
                                  yaxis = dict(range=[-3.5,3.5],),
                                  zaxis = dict(range=[-0.7,0.7],),
                                  aspectratio = dict(x=1, y=1, z=0.7),
                                      aspectmode = 'manual'))

    fig.update_yaxes(automargin=True)
    foldername = "embeddings/"+inner_folder
    filename = "embeddings/"+inner_folder+"/"+name+".png"
    if save:
        createFolder(foldername)
        fig.write_image(filename)
    if show:
        fig.show()
    return filename


# In[40]:


plot_save_fig_transform(init=[0.1, 2/3-0.1, 1/3, 0.1, 2/3-0.1, 1/3], numstep=5000, f=RPSderiv, timestep=0.1, mu=0.1, name='test', inner_folder='', show=True, save=False)


# The following code blocks have been converted to markdown, as they are used to generate the animation frames for the gif (and are hence quite time consuming to run).

# ```python
# N=200
# w1_inits = np.linspace(0.1, 1/3, N)
# 
# count=1
# image_list3 = []
# for i in w1_inits:
#     init = [i, 2/3-i, 1/3, i, 2/3-i, 1/3]
#     file_name = plot_save_fig_transform(init=init, time=1000, f=f, P=P, mu=0.1, A=5, 
#                                         name=str(count), inner_folder='combined')
#     print(str(count)+' done')
#     image_list3.append(imageio.imread(file_name))
#     count+=1
# ```

# ```python
# image_list_copy = copy.deepcopy(image_list3)
# img = image_list_copy + image_list_copy[::-1]
# imageio.mimwrite('embeddings/combined.gif', img, fps=25)
# ```

# ![Gif not found](Animations/combined2_new.gif)

# #### Actual data

# We also want to get an idea for how the actual simulation data looks with the above plotting method.

# In[41]:


def plot_save_fig_actual(init, f, name, inner_folder, timestep=0.1, numstep=1000, mu=0.1, A=5, show=False, save=True):
    data = RPSTrajectory(s=init, f=RPSderiv, numstep=numstep, timestep=timestep, mu=mu)  
    init_rounded = [round(x, 3) for x in init]

    fig = make_subplots(rows=2, cols=2, column_widths=[0.5, 0.5], row_heights=[0.2, 0.8],
        specs=[[{"type": "ternary"}, {"type": "ternary"}],
               [{"type": "scene", "colspan":2}, None]])

    fig.add_trace(go.Scatterternary(name='Initial Y', a=[init[0]], b=[init[1]], c=[init[2]], mode='markers', 
                                     marker={'symbol': 100,'color': 'green','size': 10},),row=1, col=1),
    fig.add_trace(go.Scatterternary(name='Initial W', a=[init[3]], b=[init[4]], c=[init[5]], mode='markers', 
                                     marker={'symbol': 100,'color': 'red','size': 10},),row=1, col=2),
    fig.add_trace(go.Scatter3d(x=data['w1'], y=data['w2'], z=data['y1'], mode='markers',showlegend=False,
                                  marker=dict(size=2,cmax=0.7, cmin=0.1, color=data['y2'], colorscale='rainbow',colorbar=dict(
                                      title="Y2", thickness=20, len=0.6, y=0.35)),
                                  line=dict(color='black', width=4)),row=2, col=1)

    fig.update_ternaries({
        'aaxis':{'title': 'Y1', 'min': 0.01, 'linewidth':2, 'ticks':'outside' },
        'baxis':{'title': 'Y2', 'min': 0.01, 'linewidth':2, 'ticks':'outside' },
        'caxis':{'title': 'Y3', 'min': 0.01, 'linewidth':2, 'ticks':'outside' }}, col=1)
    
    fig.update_ternaries({
        'aaxis':{'title': 'W1', 'min': 0.01, 'linewidth':2, 'ticks':'outside' },
        'baxis':{'title': 'W2', 'min': 0.01, 'linewidth':2, 'ticks':'outside' },
        'caxis':{'title': 'W3', 'min': 0.01, 'linewidth':2, 'ticks':'outside' }}, col=2)

    fig.update_layout(template="seaborn",
                      title='Inits:'+str(init_rounded),
                       width=900,
                       height=900,
                       autosize=False,
                       showlegend=True,
                       font=dict(size=15),
                       scene=dict(xaxis_title='W1',
                                  yaxis_title='W2',
                                  zaxis_title='Y1',
                                  xaxis = dict(range=[0,1]),
                                  yaxis = dict(range=[0,1],),
                                  zaxis = dict(range=[0,1],),
                                  aspectratio = dict(x=1, y=1, z=0.7),
                                      aspectmode = 'manual'))

    fig.update_yaxes(automargin=True)
    foldername = "embeddings/"+inner_folder
    filename = "embeddings/"+inner_folder+"/"+name+".png"
    if save:
        createFolder(foldername)
        fig.write_image(filename)
    if show:
        fig.show()
    return filename


# In[42]:


plot_save_fig_actual(init=[0.1, 2/3-0.1, 1/3, 0.1, 2/3-0.1, 1/3], numstep=5000, f=RPSderiv, timestep=0.1, mu=0.1, name='test', inner_folder='', show=True, save=False)


# ```python
# N=200
# w1_inits = np.linspace(0.1, 1/3, N)
# 
# count=1
# image_list4 = []
# for i in w1_inits:
#     init = [i, 2/3-i, 1/3, i, 2/3-i, 1/3]
#     file_name = plot_save_fig_actual(init=init, time=1000, f=f, P=P, mu=0.1, 
#                                      A=5, name=str(count), inner_folder='combined_actual')
#     print(str(count)+' done')
#     image_list4.append(imageio.imread(file_name))
#     count+=1
# ```

# ```python
# image_list_copy = copy.deepcopy(image_list4)
# img = image_list_copy + image_list_copy[::-1]
# imageio.mimwrite('embeddings/combined_actual.gif', img, fps=25)
# ```

# ![Gif not found](Animations/combined2_actual.gif)

# ## Large Scale Simulations

# In[47]:


im = Image.open('pikachuBW.png','r')
pix = im.load()
plt.rcParams["axes.grid"] = False
plt.imshow(im)
plt.show()


# In[48]:


sigmoid = lambda x: 1/(1 + np.exp(-5*(x-0.5)))


# In[49]:


def RPSDerivNNodes(s,t,P,mu,N, self_loops, graph, orig=False):
    """ Defines the ODE for time evolving RPS (N nodes). 
    
    Parameters:
    s (array): Array-like of initial conditions [y1, y2, y3, w1, w2, w3]
    t (int): Time to integrate function over
    P (matrix): Standard RPS payoff matrix
    mu (float): Parameters for time evolving system
    
    Returns:
    array: concatenated derivatives of all players
    """
    
    x = s.reshape(N, 3)
    A = []
    I = np.array([[1., 0., 0.], 
                   [0., 1., 0.], 
                   [0., 0., 1.]])
    
    P = np.array([[0., -1., 1.],
                   [1., 0., -1.],
                   [-1., 1., 0.] ])

    for i in range(N):
      A.append(np.matrix([[0, (x[i][1]-x[i][0]), (x[i][2]-x[i][0])],
                      [(x[i][0]-x[i][1]), 0, (x[i][2]-x[i][1])],
                      [(x[i][0]-x[i][2]), (x[i][1]-x[i][2]), 0] ]))
    utils = []
    dxdt=[]
    for i in range(N):
      util = [graph[i][j]*A[j] for j in range(len(graph[i]))]
      if self_loops[i] != 0:
        util[i]= P*self_loops[i]
      utils.append(np.sum(np.array(util), axis=0))
      dxdt_j = np.multiply(x[i], (utils[i]@x[i])-x[i].T@utils[i]@x[i]).flatten().tolist()
      dxdt.append(dxdt_j)
    
    return np.array(dxdt).flatten()

def RPSTrajectoryNnode(s, N, self_loops, graph, f=RPSDerivNNodes,
    timestep=0.1, numstep=1000, mu=0.1, P=P) :
    """ Runs ODEint for the RPS system
    
    Parameters:
    s (array): Array-like of initial conditions [x1, x2, x3, w1, w2, w3]
    timestep (float): Timestep of each iteration of the integration
    numstep (int): Number of iterations to be performed
    P (matrix): Standard RPS payoff matrix
    mu (float): Parameter for time evolving system
    
    Returns: 
    dict: Keys are (times, x1, x2, x3, w1, w2, w3)
    The values associated with the keys are time series.
    """
    x = s.flatten()
    partuple=(P, mu, N, self_loops, graph)        # Converts parameters to a tuple in the right order
    tvals=np.arange(numstep)*timestep
    traj=odeint(f,x,tvals,partuple, full_output=1)
    print(traj[1])
    return traj[0].reshape(numstep, N, 3)#[:2000]


# In[50]:


def GetEntropyVals(data, mu):
  x_weighted = []
  weight = 1
  for i in tqdm(range(data.shape[1])):
      # print(i)
      div_x = []
      for j in data[:,i]:
          kl_div_x = entropy([1/3, 1/3, 1/3], qk=j)
          div_x.append(kl_div_x)

      x_weighted.append(np.array([weight*x for x in div_x]))
      weight = weight*mu[i]
      # print(weight)
  return x_weighted

def PlotEntropyVals(entropy_vals):
    cumsum = 0
    lines=[go.Scatter(y=entropy_vals[0], mode='lines', line=dict(width=0.5), fill='tozeroy')]
    for i in range(len(entropy_vals)):
        cumsum += entropy_vals[i]
        lines.append(go.Scatter(y=cumsum, mode='lines', line=dict(width=0.5), fill='tonexty'))
    # entropy_sum = np.sum(entropy_vals, axis=0)
    lines.append(go.Scatter(y=cumsum, mode='lines',line = dict(width = 3, color='#440154'),opacity=1))
    fig = go.Figure(lines)

    # Edit layout
    fig.update_layout(title='KL-Divergence for {}-node polymatrix game'.format(len(entropy_vals)),
                      xaxis_title='Time Steps',
                      yaxis_title='KL-Divergence', 
                      # legend_orientation='h', 
                      # legend=dict( y=-0.2),
                      font=dict(size=15))
    return fig


# In[51]:


mu = [1]*64
self_loops_all = [1]*64

# indices = [2, 5, 11, 13, 15, 16, 24, 25, 34, 39, 42, 47, 52, 57, 61]
indices = [2, 3, 14, 15, 48, 49, 55, 56, 61, 62]
for index in sorted(indices, reverse=True):
    self_loops_all[index] = 0

indices = [7, 18, 33]
for index in sorted(indices, reverse=True):
    self_loops_all[index] = 2

graph2 = np.zeros((64, 64))
graph2[2][3] = -mu[2]
graph2[3][2] = 1
graph2[14][15] = -mu[14]
graph2[15][14] = 1
graph2[48][49] = -mu[48]
graph2[49][48] = 1
graph2[55][56] = -mu[55]
graph2[56][55] = 1
graph2[61][62] = -mu[61]
graph2[62][61] = 1


# Obtain a list of color values that map to a pikachu similar to the imported image above.

# In[52]:


sig_color_list = []
for j in range(8):
  for i in range(8):
    rgb_val = pix[(75+i*150, 75+j*150)]
    if rgb_val[1] == 0:
      bw_val = 0.5
    elif rgb_val[0]/255 > 0.5:
      bw_val = 0.51
    else:
      # bw_val = rgb_val[0]/255
      bw_val = 0.49
    sig_color_list.append(bw_val)
for i in range(len(sig_color_list)):
  if sig_color_list[i] < 0.3:
    sig_color_list[i] = 0.47


# In[53]:


sig_color_list = [sigmoid(x) for x in sig_color_list]
s_sig = []
for i in sig_color_list:
  init = [i, 1/3, 2/3-i]
  s_sig.append(init)
s_sig = np.array(s_sig)


# In[54]:


plt.rcParams["axes.grid"] = False
plt.imshow(np.array(sig_color_list).reshape(8,8), cmap='viridis')


# The following code is set to run for fewer iterations than we used for our experiments, in order to save time. For context, with the settings above it took over 100,000 iterations for a similar-looking Pikachu to reappear.

# In[55]:


data_64node_sig1 = RPSTrajectoryNnode(s=s_sig, numstep=2000, f=RPSDerivNNodes, mu=mu, N=64, self_loops=self_loops_all, graph=graph2)


# In[56]:


fps = 10
nSeconds = 20

# First set up the figure, the axis, and the plot element we want to animate
fig = plt.figure( figsize=(8,8) )

a = (sigmoid(data_64node_sig1[0][:,0].reshape(8,8)))
im = plt.imshow(a, cmap='viridis')
plt.axis('off')
def animate_func(i):
    if i % fps == 0:
        print( '.', end ='' )
    im.set_array(sigmoid(data_64node_sig1[i][:,0].reshape(8,8)))
    plt.axis('off')
    return [im]

anim = animation.FuncAnimation(
                               fig, 
                               animate_func, 
                               frames = nSeconds * fps,
                               interval = 1000 / fps, # in ms
                               )

# Uncomment to save the resulting animation as an mp4 file
# anim.save('animation2.mp4', fps=fps, extra_args=['-vcodec', 'libx264'])

print('Done!')


# In[57]:


HTML(anim.to_html5_video())


# ## Torus-shaped Graph with Many Nodes

# The clusters dictionary takes in 
#  - center        : list of center nodes
#  - centerconnect : lists of nodes connected to the each center
#  - leafs         : tuples with (leaf node index, [list of centers its connected to])
#  
# For example, clusters={"center":[1,7],"centerconnect":[[0,2,3,4],[4,5,6,8]],"leafs":[(0,[1]),(2,[1]),(3,[1]),(4,[1,7]),(5,[7]),(6,[7]),(8,[7])]} looks like:
# ```
# 
#     2        6
#     |        |
# 0 - 1 -  4 - 7 -  8
#     |        |
#     3        5
# 
# ```
# Instead of daisy chaining we can also tile like so:
# ```
#     12       15
#     |        |
# 9 - 10 - 11 -13- 14 
#     |        |
#     2        6
#     |        |
# 0 - 1 -  4 - 7 -  8
#     |        |
#     3        5
# 
# ```

# In[58]:


I = np.array([[1., 0., 0.], 
               [0., 1., 0.], 
               [0., 0., 1.]])

P = np.array([[0., -1., 1.],
               [1., 0., -1.],
               [-1., 1., 0.] ])
def RPSderivCluster(s,t,P,mu, Nodes, clusters):
    
    x=np.zeros((3,Nodes))
    #print(np.)
    #print(Nodes)
    for n in range(Nodes):
        for i in range(3):
            x[i,n]=s[i+3*n]
    #print(x)
    A=np.zeros((Nodes,3,3))
    for n in range(Nodes):
        #A=np.zeros((3,3))
        for i in range(3):
            for j in range(3):
                A[n,i,j]=x[j,n]-x[i,n]
    mu1, mu2, mu3, mu4 = mu


    dxsdt=[]
    center=clusters["center"]
    centerconnect=clusters["centerconnect"]
    leafs=clusters["leafs"]
    
    for n in range(Nodes):
        if n in center:
            P1w=np.zeros((3,3))
            for leaf in leafs:
                if n in leaf[1]:
                    P1w=P1w+A[leaf[0],:,:]
            x1=np.copy(x[:,n].reshape(3,1))
            dxsdt=dxsdt+np.multiply(x1, P1w@x1-x1.T@P1w@x1).flatten().tolist()

        else:
            P1w=P
            for leaf in leafs:
                if n==leaf[0]:
                    for l in leaf[1]:
                        P1w=P1w-mu1*A[l,:,:] # edit this to take in the right mu if you want different mus
            x1=np.copy(x[:,n].reshape(3,1))
            dxsdt=dxsdt+np.multiply(x1, P1w@x1-x1.T@P1w@x1).flatten().tolist()


    return dxsdt 

def RPSTrajectoryCluster(s, f=RPSderivCluster,
    timestep=0.1, numstep=1000, mu=0.1, P=P, Nodes=1,clusters={"center":[],"centerconnects":[],"leafs":[]}) :
    """ Runs ODEint for the RPS system
    
    Parameters:
    s (array): Array-like of initial conditions [x1, x2, x3, w1, w2, w3]
    timestep (float): Timestep of each iteration of the integration
    numstep (int): Number of iterations to be performed
    P (matrix): Standard RPS payoff matrix
    mu (float): Parameter for time evolving system
    
    Returns: 
    dict: Keys are (times, x1, x2, x3, w1, w2, w3)
    The values associated with the keys are time series.
    """
    partuple=(P, mu,Nodes,clusters)        # Converts parameters to a tuple in the right order
    tvals=np.arange(numstep)*timestep
    traj=odeint(f,s,tvals,partuple)

    # Store the results of odeint in a dictionary
    data={}
    data["times"]=tvals
    for n in range(Nodes):
        data[n]=traj[:,n*3:(n+1)*3]

    return data

def ani_frame(data_5node,grid_config=[9,12],numiters=1000,Nodes=9,filename='movie', dpi=100,cmap='Greys'):
    fig = plt.figure(figsize=(8,8))
    ax = fig.add_subplot(111)
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
    svals_=[]
    for n in range(Nodes):
        svals_.append(data_5node[n][0,0])
    
    svals_=np.asarray(svals_).reshape(grid_config[0],grid_config[1])
    im = ax.imshow(svals_,cmap=cmap) #,interpolation='nearest')
    #im.set_clim([0,1])
    #fig.set_size_inches([5,5])

    tight_layout()


    def update_img(n):
        #tmp = rand(300,300)
        svals_=[]
        for i in range(Nodes):
            svals_.append(data_5node[i][n,0])
    
        svals_=np.asarray(svals_).reshape(grid_config[0],grid_config[1])
        im.set_data(svals_)
        return im

    #legend(loc=0)
    ani = animation.FuncAnimation(fig,update_img,numiters,interval=30)
#     writer = animation.writers['ffmpeg'](fps=30)

#     ani.save(filename+'.mp4',writer=writer,dpi=dpi)
    return ani


# In[59]:


Nodes=100 # no plus one because of the wrapping
s3=[]
for n in range(Nodes):
    samp=np.random.rand(3); samp=(samp/sum(samp)).tolist()
    s3=s3+samp

li=[]
centers=[1]
numcenters=int(Nodes/4)
for i in range(1,numcenters):
    centers.append(4+centers[-1])
centerconnects=[[0,2,3,4]]
leafs=[(0,[1,Nodes-3]),(2,[1]),(3,[1]),(4,[1,centers[1]])]
for n in range(1,numcenters):
    temp=[centerconnects[-1][-1]]
    for i in range(1,4):
        temp.append(centers[n]+i)
        if i==3 and n<numcenters-1:
            l=(centers[n]+i,[centers[n],centers[n+1]])
        else:
            l=(centers[n]+i,[centers[n]])
        leafs.append(l)
    centerconnects.append(temp)

centerconnects[-1][-1]=0
clusters={"center":centers, "centerconnect":centerconnects, "leafs":leafs[:-1]}
# centers,centerconnects,leafs[:-1]


# In[60]:


mu3 = [1,1,1,1] 
data_100node = RPSTrajectoryCluster(s=s3, numstep=2000, f=RPSderivCluster, mu=mu3, clusters=clusters, Nodes=Nodes)

div =[]
div_combined=np.zeros(len(data_100node[0]))
divs=[]
for n in tqdm(range(Nodes)):
    div=[]
    for i in data_100node[n]:
        kl_div = entropy([1/3, 1/3, 1/3], qk=i)
        div.append(kl_div)
    x_weighted = np.asarray(div) #[x for x in div]
    div_combined=div_combined+x_weighted
    divs.append(div_combined)


# In[61]:


plots=[go.Scatter(y=divs[0], mode='lines', line=dict(width=0.5), fill='tozeroy')]
cols=[]
for i in range(1,Nodes):
    plots.append(go.Scatter(y=divs[i],
                    mode='lines', line=dict(width=0.5),#, color=np.random.choice(px.colors.sequential.Plasma)), #'#fde725'),
                    name='S'+str(i), fill='tonexty'))
plots.append(go.Scatter(y=div_combined,
                    mode='lines',
                    name='Sum of Divergences', line = dict(width = 3, color='#440154'),opacity=1))
fig = go.Figure(plots 
                )

# Edit layout
fig.update_layout(title='KL-Divergence for Torus Network',
                  xaxis_title='Time Steps',
                  yaxis_title='KL-Divergence', 
                  legend_orientation='h', 
                  legend=dict( y=-0.2),
                  showlegend=False,
                  font=dict(size=15))


# #### Animation for large number of nodes

# In[62]:


Nodes=400
grid_config=[20, 20]
s3=[]
svals=[]
cmap = plt.cm.OrRd

x_shape = 0.8*np.eye(grid_config[0])+np.fliplr(0.8*np.eye(grid_config[0]))
x_shape = x_shape.flatten().tolist()
for i in range(len(x_shape)):
    if x_shape[i] == 0.0:
        s3 += [0.1,0.1,0.8]
    else:
        s3 += [0.8,0.1,0.1]

li=[]
centers=[1]
numcenters=int(Nodes/4)
for i in range(1,numcenters):
    centers.append(4+centers[-1])
centerconnects=[[0,2,3,4]]
leafs=[(0,[1,Nodes-3]),(2,[1]),(3,[1]),(4,[1,centers[1]])]
for n in range(1,numcenters):
    temp=[centerconnects[-1][-1]]
    for i in range(1,4):
        temp.append(centers[n]+i)
        if i==3 and n<numcenters-1:
            l=(centers[n]+i,[centers[n],centers[n+1]])
        else:
            l=(centers[n]+i,[centers[n]])
        leafs.append(l)
    centerconnects.append(temp)

centerconnects[-1][-1]=0
clusters={"center":centers, "centerconnect":centerconnects, "leafs":leafs[:-1]}


# In[63]:


mu3 = [1,1,1,1] 
data_Nnode = RPSTrajectoryCluster(s=s3, numstep=2000, f=RPSderivCluster, mu=mu3, clusters=clusters, Nodes=Nodes)

div =[]
div_combined=np.zeros(len(data_Nnode[0]))
divs=[]
for n in range(Nodes):
    div=[]
    for i in data_Nnode[n]:
        kl_div = entropy([1/3, 1/3, 1/3], qk=i)
        div.append(kl_div)
    x_weighted = np.asarray(div) #[x for x in div]
    div_combined=div_combined+x_weighted
    divs.append(div_combined)


# In[64]:


plots=[go.Scatter(y=divs[0], mode='lines', line=dict(width=0.5), fill='tozeroy')]
cols=[]
for i in range(1,Nodes):
#     if i%10 == 0:
    plots.append(go.Scatter(y=divs[i],
                    mode='lines', line=dict(width=0.5),#, color=np.random.choice(px.colors.sequential.Plasma)), #'#fde725'),
                    name='S'+str(i), fill='tonexty'))
plots.append(go.Scatter(y=div_combined,
                    mode='lines',
                    name='Sum of Divergences', line = dict(width = 3, color='#440154'),opacity=1))
fig = go.Figure(plots 
                )

# Edit layout
fig.update_layout(title='KL-Divergence for Large Torus Network',
                  xaxis_title='Time Steps',
                  yaxis_title='KL-Divergence', 
                  legend_orientation='h', 
                  legend=dict( y=-0.2),
                  showlegend=False,
                  font=dict(size=15))


# In[65]:


torus_anim = ani_frame(data_Nnode,grid_config=grid_config,numiters=1000,Nodes=Nodes,filename='torusmovie', dpi=100, cmap=cmap)


# In[66]:


HTML(torus_anim.to_html5_video())

