# Weekly Report 7

**Hours spent:** 6

## What did I do this week?

This week I completed the second peer review assigned to me and also received a new peer review for my own project. Based on that feedback, I updated the PCA implementation to use the Turk and Pentland sample-space method instead of building the larger feature-space covariance matrix directly.

I also polished the executable side of the project by adding a proper installed `facerec` command and updating the usage documentation. After the changes I reran the tests, coverage, and CLI checks.

## How has the program progressed?

The project progressed from a technically finished state to a more polished final-submission state. The Eigenfaces implementation now follows the classical approach more closely, and the project also has a clearer executable entrypoint.

The test suite became stronger this week because of the new PCA-related test cases. The project currently passes 80 tests, the CLI flow works, and branch coverage remains high.

## What did I learn this week/today?

This week I learned more about the original Eigenfaces formulation and why computing eigenvectors in sample space can be more practical than working directly in the full feature space. I also learned that peer review can still lead to meaningful improvements even when a project is already working.

## What remains unclear or has been challenging?

At this point the main challenge is mostly making sure the final submission stays aligned with the latest code and peer review feedback. The technical side is in a good state, but small documentation updates may still be needed if I make final changes.

## What will I do next?

Next I will do the final repository polish for submission, update any documentation that still needs to reflect this week’s PCA change, and check once more that the final deliverables are complete.

If additional small fixes are still needed before submission, the repository and this weekly report may still be updated.
