# Autonomous racing

<p align="center">
    <img src="img/vid.gif" width="900">
</p>

Planning & control pipeline for an autonomous racing car.  
Random tracks are generated using [random-track-generator](https://github.com/mvanlobensels/random-track-generator).

## Architecture

The first lap is an exploration lap: the track is unknown, so the car only sees cones within 10 m. Track boundaries are estimated from the visible cones, a midline is fitted through them and tracked with a Stanley steering controller and a P velocity controller following a speed profile. Meanwhile all observed cones are stored in a map.

After the first lap the map is used to compute a minimum-time global trajectory, tracked my an MPC, warm-started from the global solution.

```mermaid
flowchart LR
    cones[Cones in sensor range] --> ald[Boundary estimation]
    cones --> map[(Cone map)]

    subgraph lap1[Exploration lap]
        ald --> mid[Midline generator] --> sp[Speed profile]
        mid --> stanley[Stanley steering]
        sp --> pvel[P velocity control]
    end

    subgraph laps[Subsequent laps]
        map --> gto[Global min-time trajectory] --> mpc[MPC]
    end

    stanley --> car[Vehicle]
    pvel --> car
    mpc --> car
```

## Install

```bash
uv sync
```

## Run

```bash
uv run main.py
```

