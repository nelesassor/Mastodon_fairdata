## Analysis of Fairdata in Social Media

# Project Description: Text-based Data Collection on FAIR-Data Topics from Social Media

This project aims to create a high-quality text-based dataset by collecting and analyzing posts related to FAIR-Data (Findable, Accessible, Interoperable, Reusable) principles from social media platforms. In today's data-driven research landscape, adherence to FAIR principles is crucial for enhancing research quality. Discussions and debates surrounding FAIR-Data can increase awareness of these principles and provide insights into public opinion.

The analysis of public sentiment can help us understand:

1. The progress of FAIR-Data adoption in different research domains.
2. Whether there is a predominantly positive or negative sentiment towards FAIR principles.
3. The usage of supporting resources such as learning materials or tools and identifying any existing needs.

Social media platforms like Twitter and Mastodon serve as valuable sources for capturing public opinions. The primary goal of this project is to build a qualitative text-based dataset based on contributions from these platforms. This dataset will serve as a foundation for analyzing the domain-specific discourse on FAIR-Data.

To ensure the collected data can be utilized for further research, it is essential to adhere to the research data lifecycle. This involves the following tasks:

- **Planning:** Clearly document the purpose and nature of the collected data in project documentation and data repositories.

- **Data Collection:** Evaluate relevant communication platforms for data collection. Carefully select keywords and search strategies to identify relevant posts. Consider variations in terminologies such as Research Data Management, FAIR, and Open, and address them accordingly.

- **Data Processing and Analysis:** Cleanse the data in a transparent manner. Verify that the content is relevant to FAIR principles. Classify text data to exclude non-discussion-related content, such as event announcements, from further analysis if required.

- **Data Access:** Ensure open access to the collected data, both in terms of data repositories and data formats. Consider open licensing options for maximum openness.

- **Data Archiving:** Determine appropriate data archiving methods to preserve the dataset effectively.

- **Data Reuse:** Enable efficient expansion and reuse of the dataset within the repository.

For more information about FAIR principles, please refer to [FAIR Principles](https://www.go-fair.org/fair-principles/).

This project strives to contribute to the understanding of FAIR-Data principles, their adoption in research domains, and the sentiment surrounding them. The resulting dataset will facilitate further research and analysis in the field of FAIR-Data.

The dataset can be found here: https://doi.org/10.5281/zenodo.8252443


## Project Structure

The project is organized as follows:

- **results**: Directory holding the results or output from any analysis, processing, or scripts.

- **src**: Contains the source code related to the project, be it preprocessing scripts, analysis codes, or utility functions.

- **data**: A directory for processed and perhaps some raw datasets necessary for the project.

- **licence**: Contains the details of the license under which the project is distributed.
  
- **requirements.txt**: A list of necessary packages or libraries required for the project.

- **LICENSE**: The official license file for the project.

- **CITATION.cff**: Provides citation information for the project, especially useful for academic purposes.

- **README.md**: The main README file offering an overview and introductory information about the project.

## Data to be stored and archived:

- **id**: Within a dataset, the ID allows for the unique identification of each record. This is particularly important when it comes to searching for, updating, or referencing specific information.

- **created_at**: The creation date provides a temporal context for each post and can be used for time-based analysis of discussions.

- **content**: The content of the posts is the primary focus of the analysis as it directly reflects the discussions and opinions regarding the FAIR principles.

- **account**: Information about the post's author can be helpful in analyzing discussion patterns or identifying influential accounts. This is especially relevant when conducting network analyses.

- **replies_count, reblogs_count, favourites_count**: These metrics are indicators of the response and engagement a post generates, and thus can serve as a measure of its relevance.

- **language**: The language of the posts can provide valuable insights into the regional or linguistic spread of discussions about the FAIR principles.

- **mentions, tags, emojis**: These elements can provide additional context and assist in the analysis of the content.

- **category**: Category of the toot

## Installation Instructions
1. Clone or download the project repository to your local machine.
2. Ensure you have Python 3.x installed.
3. Install the required libraries by running the following command: run `pip install -r requirements.txt`.

## License

[License](LICENSE)

## Citation 

[Citation](CITATION.cff)

## Contact Information 
nele.sassor@uni-potsdam.de





