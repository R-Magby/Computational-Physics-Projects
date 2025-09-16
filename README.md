# Computational-Physics-Projects
In this repository I share my codes and notes related to physics simulations and complex calculations.


## Collapse scalar field
Simulate the dynamics of collapse scalar field based on this article: "Gravitational collapse of k-essence"
-  I perfomr a code in C, optmized with CUDA, the numerical methods are explicit euler method for the evolution, diference fintie for the derivate, the triagonal method for the Lapso and the integrals used the Simpson's method.

## Semi-classic collapse scalar field (Thesis).
I replicated the results of the article: "Gravitational collapse of quantum fields and Choptuik scaling", also i implemet a modify the absorbing boundary conditions.
- The system consist in a massless scalar field with a quantum correction using coherent states in a spherically symmetric space-time, also employ Pauli–Villars regularization with five regulator fields and cosmological constant to handle the inherent divergences in the theory.
- The boundary condition be modify for absorb the normal modes, the tipical ABC correspond for a analict solutions for the wave equations and Klein-Gordon equations, but the Fourier solutions for a quantum field has linear combinations of incoming and outgoing waves. I porpussed a ABC than depend of mode k and mode l.
- The codes was wrote in C lenguage and optimized with CUDA, however the numerical methods are:
  - Implicit 10th Runge-Kutta, with the Gauss-Legendre cofficient, in the evolution equations.
  - 10th diference finite, for the derivate.
  -  4th Kreis-Oliger dissipation, artificial numeric
  -  And rectangular method, for the integrals.

## Wave simulations with deep learning (PINN) 
I simulated the wave equations in cartesian coordinate using deep learning, the model for this is: "Physics-Informed Neural Network".
- This project i realized because explore the potential and efficient the PINN.
- The similation was 
