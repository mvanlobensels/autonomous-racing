"""
Simulation script to test the nonlinear dynamic bicycle model.

This script runs several test scenarios and plots the results to validate
the vehicle dynamics model behavior.
"""

import numpy as np
import matplotlib.pyplot as plt
from bicycle_model import NonlinearBicycleModel, VehicleParameters


def simulate_step_steer(model: NonlinearBicycleModel, v_initial: float = 20.0,
                        delta: float = 0.1, duration: float = 5.0, dt: float = 0.01):
    """
    Simulate a step steering input at constant speed.

    Args:
        model: Bicycle model instance
        v_initial: Initial velocity [m/s]
        delta: Steering angle [rad]
        duration: Simulation duration [s]
        dt: Time step [s]

    Returns:
        Dictionary containing time history of states and controls
    """
    # Initialize state: [x, y, psi, v_x, v_y, omega]
    state = np.array([0.0, 0.0, 0.0, v_initial, 0.0, 0.0])

    # Time vector
    t = np.arange(0, duration, dt)
    n_steps = len(t)

    # Storage arrays
    states = np.zeros((n_steps, 6))
    controls = np.zeros((n_steps, 2))

    # Simulation loop
    for i in range(n_steps):
        # Store current state
        states[i] = state

        # Control: maintain speed with throttle, apply constant steering
        # Simple speed controller
        throttle = 0.5 if state[3] < v_initial else 0.3
        control = np.array([delta, throttle])
        controls[i] = control

        # Step forward
        state = model.step(state, control, dt)

    return {
        't': t,
        'states': states,
        'controls': controls
    }


def simulate_lane_change(model: NonlinearBicycleModel, v_initial: float = 20.0,
                        duration: float = 6.0, dt: float = 0.01):
    """
    Simulate a double lane change maneuver.

    Args:
        model: Bicycle model instance
        v_initial: Initial velocity [m/s]
        duration: Simulation duration [s]
        dt: Time step [s]

    Returns:
        Dictionary containing time history of states and controls
    """
    # Initialize state
    state = np.array([0.0, 0.0, 0.0, v_initial, 0.0, 0.0])

    # Time vector
    t = np.arange(0, duration, dt)
    n_steps = len(t)

    # Storage arrays
    states = np.zeros((n_steps, 6))
    controls = np.zeros((n_steps, 2))

    # Simulation loop
    for i, time in enumerate(t):
        # Store current state
        states[i] = state

        # Double lane change steering profile
        if time < 1.0:
            delta = 0.0
        elif time < 2.0:
            delta = 0.15 * np.sin(np.pi * (time - 1.0))
        elif time < 3.5:
            delta = -0.15 * np.sin(np.pi * (time - 2.0) / 1.5)
        else:
            delta = 0.0

        # Simple speed controller
        throttle = 0.5 if state[3] < v_initial else 0.3
        control = np.array([delta, throttle])
        controls[i] = control

        # Step forward
        state = model.step(state, control, dt)

    return {
        't': t,
        'states': states,
        'controls': controls
    }


def simulate_acceleration(model: NonlinearBicycleModel, duration: float = 10.0,
                         dt: float = 0.01):
    """
    Simulate straight-line acceleration from standstill.

    Args:
        model: Bicycle model instance
        duration: Simulation duration [s]
        dt: Time step [s]

    Returns:
        Dictionary containing time history of states and controls
    """
    # Initialize state at rest
    state = np.array([0.0, 0.0, 0.0, 0.1, 0.0, 0.0])

    # Time vector
    t = np.arange(0, duration, dt)
    n_steps = len(t)

    # Storage arrays
    states = np.zeros((n_steps, 6))
    controls = np.zeros((n_steps, 2))

    # Simulation loop
    for i, time in enumerate(t):
        # Store current state
        states[i] = state

        # Full throttle, no steering
        if time < 8.0:
            throttle = 1.0
        else:
            throttle = 0.0  # Coast

        control = np.array([0.0, throttle])
        controls[i] = control

        # Step forward
        state = model.step(state, control, dt)

    return {
        't': t,
        'states': states,
        'controls': controls
    }


def plot_results(results: dict, title: str):
    """
    Plot all states and control inputs over time.

    Args:
        results: Dictionary containing simulation results
        title: Title for the figure
    """
    fig, axes = plt.subplots(4, 2, figsize=(14, 12))
    fig.suptitle(title, fontsize=14, fontweight='bold')

    t = results['t']
    states = results['states']
    controls = results['controls']

    # Plot 1: XY trajectory
    axes[0, 0].plot(states[:, 0], states[:, 1], 'b-', linewidth=2)
    axes[0, 0].plot(states[0, 0], states[0, 1], 'go', markersize=8, label='Start')
    axes[0, 0].plot(states[-1, 0], states[-1, 1], 'ro', markersize=8, label='End')
    axes[0, 0].set_xlabel('X Position [m]')
    axes[0, 0].set_ylabel('Y Position [m]')
    axes[0, 0].set_title('Vehicle Trajectory')
    axes[0, 0].grid(True)
    axes[0, 0].axis('equal')
    axes[0, 0].legend()

    # Plot 2: Yaw angle
    axes[0, 1].plot(t, np.rad2deg(states[:, 2]), 'g-', linewidth=2)
    axes[0, 1].set_xlabel('Time [s]')
    axes[0, 1].set_ylabel('Yaw Angle [deg]')
    axes[0, 1].set_title('Yaw Angle $\\psi$')
    axes[0, 1].grid(True)

    # Plot 3: Longitudinal velocity
    axes[1, 0].plot(t, states[:, 3], 'm-', linewidth=2)
    axes[1, 0].set_xlabel('Time [s]')
    axes[1, 0].set_ylabel('$v_x$ [m/s]')
    axes[1, 0].set_title('Longitudinal Velocity')
    axes[1, 0].grid(True)

    # Plot 4: Lateral velocity
    axes[1, 1].plot(t, states[:, 4], 'c-', linewidth=2)
    axes[1, 1].set_xlabel('Time [s]')
    axes[1, 1].set_ylabel('$v_y$ [m/s]')
    axes[1, 1].set_title('Lateral Velocity')
    axes[1, 1].grid(True)

    # Plot 5: Yaw rate
    axes[2, 0].plot(t, np.rad2deg(states[:, 5]), 'orange', linewidth=2)
    axes[2, 0].set_xlabel('Time [s]')
    axes[2, 0].set_ylabel('Yaw Rate [deg/s]')
    axes[2, 0].set_title('Yaw Rate $\\omega$')
    axes[2, 0].grid(True)

    # Plot 6: Speed (magnitude)
    speed = np.sqrt(states[:, 3]**2 + states[:, 4]**2)
    axes[2, 1].plot(t, speed, 'r-', linewidth=2)
    axes[2, 1].set_xlabel('Time [s]')
    axes[2, 1].set_ylabel('Speed [m/s]')
    axes[2, 1].set_title('Total Speed')
    axes[2, 1].grid(True)

    # Plot 7: Steering angle (control input)
    axes[3, 0].plot(t, np.rad2deg(controls[:, 0]), 'k-', linewidth=2)
    axes[3, 0].set_xlabel('Time [s]')
    axes[3, 0].set_ylabel('Steering Angle [deg]')
    axes[3, 0].set_title('Steering Input $\\delta$')
    axes[3, 0].grid(True)

    # Plot 8: Throttle (control input)
    axes[3, 1].plot(t, controls[:, 1], 'purple', linewidth=2)
    axes[3, 1].set_xlabel('Time [s]')
    axes[3, 1].set_ylabel('Throttle [-]')
    axes[3, 1].set_title('Throttle Input')
    axes[3, 1].set_ylim([-0.1, 1.1])
    axes[3, 1].grid(True)

    plt.tight_layout()
    return fig


def main():
    """Run all simulations and create plots."""
    print("Initializing nonlinear bicycle model...")
    model = NonlinearBicycleModel()

    print("\n=== Running Simulations ===\n")

    # Test 1: Step steer input
    print("1. Step steering input test (v=20 m/s, delta=0.1 rad)...")
    results_step = simulate_step_steer(model, v_initial=20.0, delta=0.1)
    print(f"   Final position: ({results_step['states'][-1, 0]:.2f}, {results_step['states'][-1, 1]:.2f}) m")
    print(f"   Final yaw angle: {np.rad2deg(results_step['states'][-1, 2]):.2f} deg")

    # Test 2: Lane change maneuver
    print("\n2. Double lane change maneuver (v=20 m/s)...")
    results_lane = simulate_lane_change(model, v_initial=20.0)
    print(f"   Maximum lateral displacement: {np.max(np.abs(results_lane['states'][:, 1])):.2f} m")
    max_ay = np.max(np.abs(results_lane['states'][:, 3] * results_lane['states'][:, 5]))
    print(f"   Maximum lateral acceleration: {max_ay:.2f} m/s^2")

    # Test 3: Acceleration test
    print("\n3. Straight-line acceleration test...")
    results_accel = simulate_acceleration(model)
    print(f"   Final velocity: {results_accel['states'][-1, 3]:.2f} m/s")
    print(f"   Distance traveled: {results_accel['states'][-1, 0]:.2f} m")
    print(f"   0-100 km/h time: ", end="")
    v_kmh = results_accel['states'][:, 3] * 3.6
    idx_100 = np.where(v_kmh >= 100)[0]
    if len(idx_100) > 0:
        print(f"{results_accel['t'][idx_100[0]]:.2f} s")
    else:
        print("Not reached")

    print("\n=== Generating Plots ===\n")

    # Create plots with unified format
    plot_results(results_step, 'Step Steering Input Response')
    plot_results(results_lane, 'Double Lane Change Maneuver')
    plot_results(results_accel, 'Straight-Line Acceleration')

    print("All simulations complete!")
    print("Close the plot windows to exit.")

    plt.show()


if __name__ == "__main__":
    main()
