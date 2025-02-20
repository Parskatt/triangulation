# Benchmarking Triangulation Algos

Here's a quick comparison between niter2 (an iterative algo for computing reprojection-based MLE 2D observations, which we then triangulate) and midpoint.
Note that in both cases we use midpoint for triangulating the 3D point.

## Benchmark setup
All noise is Normally distributed.
We consider two calibrated cameras with arbitrary positions (std=1) and rotations.
We consider a 3D point with arbitrary position (std=10).
We add noise to the camera positions, camera rotations (in the Lie algebra), and observations.

## Possible improvements to "Optimal" triangulation
My thought was that triangulation is poorly conditioned when the angle between the cameras principal axis and the ray of the 3D point is ~90 degrees.
I therefore made a normalization which creates a virtual camera where the principal axis is aligned with the observation.
I found that this in general gives results that are better, but is still sometimes unstable.
I'm not sure why it is unstable.

## Results
In general results are approximately:

|                     | Mean error                            | Median error |
|---------------------|---------------------------------------|--------------|
| Niter2              | 100%                                  | 100%         |
| Niter2 (Normalized) | 60%-150% (both are somewhat unstable) | 90%          |
| Midpoint            | 80%                                   | 90%          |

## Contributing

It's likely that I've made multiple mistakes in implementation etc.
Please make an issue/PR to fix :D
