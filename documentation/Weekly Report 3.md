# Weekly Report 3

**Hours spent:** 21

## What did I do this week?

This week I continued developing the core area of the project and made the program runnable in a more complete way. I implemented nearest-neighbor classification, connected it with PCA and preprocessing, and added model saving/loading.

I also implemented command-line support for training, prediction, and evaluation. The evaluation command now writes a report with basic metrics and a confusion matrix.

At the same time, I continued developing tests alongside the code. I expanded unit tests with more representative edge cases, improved branch coverage, and started writing the Testing Document with current test and coverage details.

## How has the program progressed?

The project has moved clearly forward this week, The core functionality is now observable and runnable through CLI commands, and the main suporting methods for the current scope are in place.

Testing is also in a much stronger state now. The test suite has grown, and coverage tracking is integrated into the workflow in a practical way.

## What did I learn this week/today?

This week I learned more about how to connect separate algorithm modules into a coherent pipeline without making the code significantly harder to test. I also learned how to design better edge-case tests for branch coverage instead of only testing normal input paths.

In addition, I learned more about using static analysis with Pylint to improve consistency in naming and code structure while keeping functionality unchanged.

## What remains unclear or has been challenging?

The most challenging part this week was balancing fast progress with maintaining code quality and documentation at the same time. I feel like it is quite easy to move quickly on features, but it takes extra care to keep tests, linting, and documents aligned with each change.

Another open point is still how to best evaluate recognition quality on larger and more realistic datasets beyond the current synthetic-focused unit test setup.

## What will I do next?

Next I will continue preparing the project for peer review by improving robustness and adding relevant additional tests where needed. I will also keep updating the testing and implementation documentation as the project evolves.

After that, I will focus on finishing the remaining core functionality and improving overall polish and usability of the current CLI workflow.