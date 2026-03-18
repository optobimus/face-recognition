# Weekly Report 2

**Hours spent:** 8

## What did I do this week?

This week I started implementing the core area of the project. I created the first actual modules for the face recognition pipeline, including dataset handling and image preprocessing.

I implemented ORL-style dataset discovery and deterministic train/test splitting. I also implemented preprocessing steps for the images (grayscale conversion, resizing, normalization, and flattening). After that, I started the actual recognition core by adding PCA-based dimensionality reduction.

At the same time, I wrote unit tests for the implemented functionality and set up branch coverage tracking so testing is part of the workflow from the beginning.

## How has the program progressed?

The project has now moved from planning to concrete implementation. The basic technical structure exists, and the first core algorithms are already in place.

The codebase now has an initial facialrecognition pipeline foundation that can be extended in the following weeks with nearest-neighbor classification, end-to-end training/prediction flow, and broader evaluation.

## What did I learn this week/today?

This week I learned more about how to organize the project into small and testable modules so development and testing can progress together. I also learned more about practical details of PCA implementation and how to validate it with unit tests.

In addition, I learned how to track branch coverage and use it to identify which parts of the code still need stronger test coverage.

## What remains unclear or has been challenging?

The most challenging part so far for me has been making sure the implementation details and testing workflow match the course requirements at the same time.

Another thing that is still somewhat open is tuning the later pipeline parameters (such as component count and classification settings) to get stable recognition performance once the full pipeline is connected. But I guess I will see what the optimal way to go about this is once I get further into the project.

## What will I do next?

Next I will continue implementing the core functionality by adding nearest-neighbor classification and connecting the current parts into a complete train/predict pipeline.

After that, I will expand evaluation support and continue improving test coverage so that progress stays measurable and easy to verify.