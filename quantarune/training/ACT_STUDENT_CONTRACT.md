# ACT Student Contract

This file defines the intended **deployable student** interface for the ACT lane.

## Goal

The student policy should keep only inputs that are plausibly available at deployment time.

It should **not** depend on appended privileged `scene.*` metadata such as:

- exact NIC presence map
- exact NIC translations
- exact NIC yaw
- exact SC port presence / translation / yaw
- exact target translation / yaw
- exact task-board pose

## Student inputs

The student keeps:

- all three RGB cameras
- the base `observation.state` vector only

The current base state is `48` dims and already contains:

- TCP pose
- TCP velocity
- TCP error
- joint positions
- joint velocities
- force / torque wrench
- task fields:
  - target validity
  - cable / plug / port type ids
  - target module id
  - target port id
  - target module index
  - target port index
  - time limit

So the student still knows the legitimate task identity. It simply does not get free scene geometry.

## Teacher vs student

- Teacher: `97`-dim state + 3 cameras
- Student: `48`-dim state + 3 cameras

Both currently keep the same `19`-dim action contract until the deployment action interface is narrowed further.
