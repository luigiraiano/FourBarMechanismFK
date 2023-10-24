#!/usr/bin/env python
# Luigi Raiano
# Forward Kinematics of a 4-Bar Planar Mechanism using Netwon Raphson-Method

# Import numpy for vector handling - If not installed: pip install numpy
import numpy as np

# Import Pandas for data handling
import pandas as pd

# Import Matplot Lib - If not installed: pip install matplotlib
import matplotlib.pyplot as plot

# Import Scipy for data integration
from scipy.integrate import solve_ivp

import yaml
# Example from internet
# vdp1 = lambda T,Y: [Y[1], (1 - Y[0]**2) * Y[1] - Y[0]]
# sol = solve_ivp (vdp1, [0, 20], [2, 0])

# T = sol.t
# Y = sol.y
# https://docs.scipy.org/doc/scipy/tutorial/integrate.html#ordinary-differential-equations-solve-ivp

class FourBarMechanism:
    # Attributes
    n_link = 4
    lenghts = np.array([])
    init_angles = np.array([])
    linear_motion_range = np.array([])
    linear_motion_resolution = 0
    driver_speed = 0
    theta = np.array([])

    
    def __init__(self, n_link = 4):
        self.n_link = n_link

        for i in range(0, n_link):
            self.lenghts = np.append(self.lenghts, 0)
            self.init_angles = np.append(self.init_angles, 0)

        self.theta.resize(2)
        self.theta[0] = self.init_angles[0]
        self.theta[1] = self.init_angles[3]

    def __str__(self):
        return "Number of Links: {0} - Link Lengths: {1} - Inital Angles: {2} - Motion Range: {3} - Motion Speed: {4}".format(self.n_link, 
                                                                                                                        self.lenghts, 
                                                                                                                        np.degrees(self.init_angles), 
                                                                                                                        np.degrees(self.linear_motion_range),
                                                                                                                        np.degrees(self.driver_speed))
    
    
    def getParamsFromYaml(self, file_path):
        with open(file_path, "r") as stream:
            try:
                config_file_content = yaml.safe_load(stream)
                self.lenghts = np.array(config_file_content['lenghts'])
                self.init_angles = np.radians(np.array(config_file_content['angles']))
                self.linear_motion_range = np.radians(np.array(config_file_content['linear_driver']['motion_range']))
                self.linear_motion_resolution = np.radians(config_file_content['linear_driver']['resolution'])
                self.driver_speed = np.radians(config_file_content['linear_driver']['speed'])
            except yaml.YAMLError as exc:
                print(exc)
    
    def setLenghts(self, lengths):
        self.lenghts = np.array(lengths)

    def setInitialConfiguration(self, init_angles):
        self.init_angles = init_angles

    def setIntegrationDuration(self, duration):
        self.integration_duration = duration

    def setIntegrationSamplingTime(self, sampling_time):
        self.integration_sampling_time = sampling_time

    def getNumberOfLinks(self):
        return self.n_link
    
    def mechanismEquations(self, phi):
        # phi is in radians
        # phi = [theta2, theta3]

        y = np.arrray([])

        y[0] = self.lenghts[0]*np.cos(self.theta(0)) + self.lenghts[1]*np.cos(phi(0)) + self.lenghts[2]*np.cos(phi(1)) + self.lenghts[3]*np.cos(self.theta(1))
        y[1] = self.lenghts[0]*np.sin(self.theta(0)) + self.lenghts[1]*np.sin(phi(0)) + self.lenghts[2]*np.sin(phi(1)) + self.lenghts[3]*np.sin(self.theta(1))

        return y
    
    def mechanismJacobian(self, phi):
        # phi is in radians
        # phi = [theta2, theta3]

        J = np.matrix([])

        J[0,0] = -self.lenghts[1]*np.sin(phi[0])
        J[0,1] = -self.lenghts[2]*np.sin(phi[1])
        J[1,0] = self.lenghts[1]*np.cos(phi[0])
        J[1,1] = self.lenghts[2]*np.cos(phi[1])

        return J

    def computeKineamtics(self):
        samples = np.linspace(self.linear_motion_range[0], self.linear_motion_range[1], np.round(1/self.linear_motion_resolution))

        # Start Loop
        for i in range(0, len(samples)):
            self.theta[0] = samples[i]

        

class NewtonRaphson:
    # Attributes
    tol = 0
    iteration_max = 0

    def __init__(self, tolerance = 0.001, max_iterations = 100):
        self.tol = tolerance
        self.iteration_max = max_iterations

    def feval(funcName, *args):
        return eval(funcName)(*args)

    def solve(self, f, J, x_0):
        x = np.array([])
        x.resize(len(x_0))

        it = 0
        err = np.norm(x - x_0)

        while(err > self.tol):
            func = self.feval(f, x_0)
            Jac = self.feval(J, x_0)

        return x


def main():

    print("Object")
    mechanism = FourBarMechanism()
    print(mechanism)

    print("Get Params")
    mechanism_config_file_path = "config/4bar_mechanism_config.yaml"
    mechanism.getParamsFromYaml(file_path=mechanism_config_file_path)
    print(mechanism)


if __name__ == "__main__":
    if False:
        import cProfile
        cProfile.run('main()', 'ik_prof')
    else:
        main()