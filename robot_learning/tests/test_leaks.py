import jax
import jax.numpy as jnp

# BAD EXAMPLE - Actually shows a tracer leak
class BadRobot:
    def __init__(self):
        self.position = None
        self.dt = 0.01
    
    def reset(self):
        self.position = jnp.array([0., 0., 0.])
        return self.position
    
    def move(self, velocity):
        # BAD: Storing the traced value as instance attribute
        self.position = self.position + velocity * self.dt
        return self.position

# GOOD EXAMPLE - Proper functional approach
class GoodRobot:
    def __init__(self):
        self.initial_position = jnp.array([0., 0., 0.])
        self.dt = 0.01
    
    def reset(self):
        return self.initial_position
    
    def move(self, position, velocity):
        # GOOD: Pure function that returns new state
        return position + velocity * self.dt

# Demonstrate the leak
def main():
    # Bad example with tracer leak
    print("Testing BadRobot:")
    with jax.checking_leaks():
        bad_robot = BadRobot()
        
        # vmap will cause tracers to be created
        batch_move = jax.vmap(lambda v: bad_robot.move(v))
        
        velocities = jnp.array([[1., 0., 0.],
                               [0., 1., 0.],
                               [0., 0., 1.]])
        try:
            bad_robot.reset()
            result = batch_move(velocities)
            print("Result (shouldn't get here):", result)
        except Exception as e:
            print("Bad example error:", e)
    
    print("\nTesting GoodRobot:")
    with jax.checking_leaks():
        good_robot = GoodRobot()
        
        # Proper functional approach works with vmap
        batch_move = jax.vmap(lambda pos, v: good_robot.move(pos, v))
        
        positions = jnp.repeat(good_robot.reset()[None], 3, axis=0)
        velocities = jnp.array([[1., 0., 0.],
                               [0., 1., 0.],
                               [0., 0., 1.]])
        
        result = batch_move(positions, velocities)
        print("Good example result:", result)

if __name__ == "__main__":
    main()