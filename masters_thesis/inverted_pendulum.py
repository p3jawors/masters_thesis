# from https://colab.research.google.com/drive/1Jpje3UWRr-ZvVKR1SlcwoM-KI1OGBNYA#scrollTo=sis3Dnog4Qe0

# The code is designed to work with Python 3.x
# To provide Python 2.7 compatibility, we can use two imports:
from __future__ import print_function
from __future__ import division

# Set to True if we are in colaboratory of False elsewere
colaboratory = False

# if colaboratory:
#     # We also load the calc module
#     !rm calc.*
#     !wget https://raw.githubusercontent.com/R6500/Python-bits/master/Modules/calc.py

# We import numpy and calc
import numpy as np
import calc

# Erase output after import
from IPython.display import clear_output
clear_output()

# Check loaded modules
try:
    print('calc version: ',calc.version)
except:
    print('Error loading the calc module')

#========================================
# Data for the pendulum

m = 0.1   # [kg]   Mass
l = 0.1   # [m]    Length
g = 9.81  # [s^-2] Gravity acceleration

# Torque vs angle graph

vAngle = np.arange(-180.0,180.0,1.0)
vTl    = m*g*l*np.sin(np.deg2rad(vAngle))

# Indicate the calc module that we operate inside colaboratory
# This is needed to give more pleasant graphipcs
calc.setColaboratory(colaboratory)

# Torque graph
calc.plot11(vAngle,vTl,"Torque vs Angle","Angle (deg)","Torque (n*m)")
#========================================
# Motor data

# We will calculate the motor parameters from typical motor information
# We will only consider the intertial momentum of the pendulum

dcVoltage    = 12     # [V]
stallCurrent = 85     # [A]
stallTorque  = 0.5    # [N*m]
unloadedRPM  = 19300  # [min^-1]

# Calculations of constants of the motor

# Convert unloaded speed from RPM to rad/s
unloadedSpeed = 2.0*np.pi*unloadedRPM/60.0

# Calculate the winding resistance from the stall current
R = dcVoltage/stallCurrent

# Calculate the motor constant
k = stallTorque/stallCurrent

# Inertial momentum
J = m*l*l

# Show the results
calc.printVar("R",R,"Ohms")
calc.printVar("k",k,"N*m / A")
calc.printVar("J",J,"kg*m^2",sci=False)
#========================================
# Define the motor voltage
V = 1.0 #[V]

# Stall motor torque
Tm = V*k/R
vTm=Tm*np.ones(len(vAngle))

# Torque graph

calc.plot1n(vAngle,[vTl,vTm],"Torque vs Angle","Angle (deg)"
             ,"Torque (n*m)",["Pendulum","Motor"])
#========================================
# System dynamics

# We will consider a friction value to damp the system
mu = 2e-3 # [N*m*s]

# Time information
tEnd  = 5.0 #[s]
tStep = 0.001 #[s]
vTime = np.arange(0.0,tEnd,tStep)

# Start condition
startAngle = 170

# State variables
angle = np.deg2rad(startAngle) #rad
speed = 0 #rad/s

# Output vectors
vSpeed  = []
vAngle  = []
vTorque = []

# Do the simulation
# We solve using the Euler method because is easier to
# understand and we only need qualitative results
for time in vTime:
    # Calculate torque
    T = m*g*l*np.sin(angle)+(k/R)*(-V-k*speed)-mu*speed
    # Calculate acceleration, speed and angle
    alpha = T/J
    speedNew = speed + alpha*tStep
    angleNew = angle + speed*tStep
    # Update output data
    vTorque.append(T)
    vSpeed.append(speed)
    vAngle.append(angle)
    # Udate state information
    speed = speedNew
    angle = angleNew

# Show angle graph
calc.plot11(vTime,np.rad2deg(vAngle),"Angle in open loop"
            ,"Time (s)","Angle (deg)")
#========================================
# Closed loop operation

# This time we will solve the dynamics using Runge-Kutta
# That gives a much more exact solution depending on the
# P, I and D settings of the controller

# Some interesting cases
# P =  5, I =   0, D =  0 Proportional only (Error depends on goal angle)
# P = 20, I =   0, D =  0 Less error but more oscillations
# P =  5, I = 0.5, D =  0 Less error than P=5 only
# P =  5, I = 0.5, D =  5 Reduced oscillations
# P = 50, I =  10, D = 20 Fast and damped

# Goal angle
angleGoal = 20.0

# Proportional constant
P = 50.0

# Integral constant
I = 0.0

# Derivative constant
D = 0.0

# We will consider a friction value to damp the system
mu = 2e-3 # [N*m*s]

# Time information
# tEnd  = 5.0 #[s]
tEnd  = 30.0 #[s]
tStep = 0.001 #[s]
vTime = np.arange(0.0,tEnd,tStep)

# Start condition
startAngle = -70

# State variables [speed,angle]
x = [0.0,np.deg2rad(startAngle)]

# Output vector
vAngle  = []

# Angle goal in rad
goal = np.deg2rad(angleGoal)

# Initialize integral
integ = 0.0

# Initialize derivative
prevError = 0

# Derivative of the state variables (for Runge-Kutta)
def fderClose(x,t,u):
    global integ,prevError
    error = x[1]-goal

    # Calculate integral term
    integ = integ + error*tStep

    # Calculate voltage
    V = -(P*error+I*integ+D*(error-prevError)/tStep)
    if V > dcVoltage:
       V = dcVoltage
    if V < -dcVoltage:
       V = -dcVoltage

    prevError = error

    # Calculate torque
    T = m*g*l*np.sin(x[1])+(k/R)*(V-k*x[0])-mu*x[0]
    T += u
    # Calculate acceleration
    alpha = T/J
    # Return state derivatives
    # [0] d speed / dt = alpha
    # [1] d angle / dt = speed
    return np.array([alpha,x[0]]), T
    # print(np.array([alpha,x[0]]))
    # return np.array([alpha,x[0]])

# Do the simulation using Runge-Kutta (4th order)
u = 0
u_track = []
velAngle_track = []
for tt, time in enumerate(vTime):
    # Store data
    vAngle.append(x[1])
    if time%5 == 0:
        u += 5

    # Update state using Runge-Kutta
    x, u = calc.rk4(x,time,fderClose,tStep, u)
    u_track.append(u)
    if tt == 0:
        velAngle_track.append(0)
    else:
        velAngle_track.append((x[1]-vAngle[tt-1])/tStep)


# Goal value
vGoal = angleGoal*np.ones(len(vTime))

# Show graphs
calc.plot1n(vTime,[np.rad2deg(vAngle),vGoal]
             ,"PID operation"
             ,"Time (s)","Angle (deg)")

import matplotlib.pyplot as plt
plt.figure()
plt.subplot(311)
plt.title('Angle')
plt.plot(vTime, vAngle)
plt.subplot(312)
plt.title('Angular vel')
plt.plot(vTime, velAngle_track)
plt.subplot(313)
plt.title('Control')
plt.plot(vTime, u_track)
plt.show()

from abr_analyze import DataHandler
dat = DataHandler('codebase_test_set', 'data/databases')
dat.save(
    save_location='inverted_pendulum_pid',
    data={'dt': tStep, 'time': vTime, 'state': vAngle, 'vel': velAngle_track, 'control': u_track},
    overwrite=True
)
