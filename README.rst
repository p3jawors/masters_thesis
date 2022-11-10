***********
Learned Legendre Predictive State Estimator for Control
***********
Abstract
========
`Thesis link <http://hdl.handle.net/10012/18791>`_

This thesis introduces a novel method for system model identification, specifically for state estimation. The method uses a 2 or 3 layer neural network developed and trained with the methods of the Neural Engineering Framework (NEF). Using the NEF allows for direct control of what the different layers represent with white-box modelling of the layers. NEF networks also have the added benefit of being compilable onto neuromorphic hardware, which can run on an order of magnitude or more less power than conventional computing hardware. The first layer of the network is optional and uses a Legendre Delay Network (LDN). The LDN implements a linear operation that performs a mathematically optimal compression of a time series of data, which in this context is the input signal to the network. This allows for temporal information to be encoded and passed into the network. The LDN frames the problem of memory as delaying a signal by some length θ seconds. Using the linear transfer function for a continuous-time delay, F(s) = e−θs, the LDN compression is considered optimal as it uses Pad´e approximants to represent the delay, which has been proven optimal for this purpose. The LDN has been shown to outperform other memory cells, such as long short-term memory (LSTM) and gated recurrent units (GRU), by several orders of magnitude, and is capable of representing over 1,000,000 timesteps of data. The LDN forms a polynomial representation of a sliding window of length θ, allowing for a continuous representation of the time series. The second layer uses the Learned Legendre Predictor (LLP) to make predictions of how a subset of the input signal to this layer will evolve over a future window of time. In the case of model estimation, using the system states and control signal (at minimum), the LLP layer predicts how the system states will evolve over a continuous window into the future. The LLP uses a similar time series compression as the LDN, but of the representation of the layer prediction into the future. The weights for the LLP layer can be trained online or offline. The third layer of the network performs the transformation out of the Legendre domain into the units of the input signal to be predicted. Since the second layer outputs a polynomial representation of the state prediction, the state at any time in the prediction window can be extracted with a linear operation. Combined, the three layer network is referred to as the Learned Legendre Predictive State Estimator (LLPSE). The 2 layer version, without LDN context encoding, is tested online on a single link inverted pendulum and is able to predict the angle of the arm 30 timesteps into the future while learning the system dynamics online. The 3 layer LLPSE is trained offline to predict the future position of a simulated quadrotor over a continuous window of 1 second in length. The training, validation, and test data is generated in AirSim with Unreal Engine 4. The LLPSE is able to predict the future second of a simulated quadrotor’s position with an average RMSE of 0.0067 on the network’s normalized representation space of position (normalized from a 30x30x15 meter volume). Future work is discussed, with initial steps provided for using the LLPSE for model predictive control (MPC). A controller, the Learned Legendre Predictive Controller (LLPC), is designed and tested for state estimation across the control space. The design and future steps of the LLPC are discussed in the final chapter.

Installation
============
The install script will create an anaconda environment with python version 3.8.
It will also optionally isntall Airsim and Unreal Engine4.

To install base packages::
    ./install.sh

To install all packages::
    ./install.sh all

To install base packages and Airsim::
    ./install.sh airsim

To install base packages and UE4::
    ./install.sh ue4

.. abr-analyze           0.1.0.dev0  /home/pjaworsk/src/abr_analyze
.. abr-control           0.1.0       /home/pjaworsk/src/abr_control
.. learn-dyn-sys         0.0.2       /home/pjaworsk/src/masters/learn-dyn-sys
.. masters-thesis        0.0.1       /home/pjaworsk/src/masters/masters_thesis
.. nengo-control         0.1.0.dev0  /home/pjaworsk/src/nengo-control
.. nengo-interfaces      0.1.0.dev0  /home/pjaworsk/src/nengo-interfaces

Usage
=====
WIP
