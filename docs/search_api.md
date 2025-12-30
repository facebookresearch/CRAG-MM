# CRAG-MM Search APIs

CRAG-MM is a visual question-answering benchmark that focuses on factuality of Retrieval Augmented Generation (RAG). It offers a unique collection of image and question-answering sets to enable comprehensive assessment of wearable devices.

## CRAG-MM Search API Description

The CRAG-MM Search API is a Python library that provides a unified interface for searching images and text. It supports both image and text queries, and can be used to retrieve relevant information from a given set of retrieved contents.

The *image search* API uses CLIP embeddings to encode the images. It takes an image or an image url as input and returns a list of similar images with the relevant information about the entities contained in the image. Similarity is determined by cosine similarity of the embeddings. See *Search for images* below for an example.

The *web search* API uses chromadb full text search to build an index for pre-fetched web search results. It takes a text query as input, and returns relevant webpage urls and meta data such as page title and page snippets. You can download the webpage content based on the urls and use the information to build Retreival Augmented Generation (RAG) systems. Here, relevancy of the webpages are calculated based on cosine similarity. See *Search for text queries* below for an example.

## Installation

```bash
pip install cragmm-search-pipeline==0.5.1
```

## Usage

### Task 1
```python
from cragmm_search.search import UnifiedSearchPipeline

# initiate image search API only, web search API is not enabled for Task 1
## validation
search_pipeline = UnifiedSearchPipeline(
    image_model_name="openai/clip-vit-large-patch14-336",
    image_hf_dataset_id="crag-mm-2025/image-search-index-public-test",
)

```

### Task 2 & 3

```python
from cragmm_search.search import UnifiedSearchPipeline

# initiate both image and web search API
## validation
search_pipeline = UnifiedSearchPipeline(
    image_model_name="openai/clip-vit-large-patch14-336",
    image_hf_dataset_id="crag-mm-2025/image-search-index-public-test",
    text_model_name="BAAI/bge-large-en-v1.5",
    web_hf_dataset_id="crag-mm-2025/web-search-index-public-test",
)

## public_test
search_pipeline = UnifiedSearchPipeline(
    image_model_name="openai/clip-vit-large-patch14-336",
    image_hf_dataset_id="crag-mm-2025/image-search-index-public-test",
    text_model_name="BAAI/bge-large-en-v1.5",
    web_hf_dataset_id="crag-mm-2025/web-search-index-public-test",
)


# optional, can specify the tag of the index. default is "main". we recommend always use default / "main".
search_pipeline = UnifiedSearchPipeline(
    image_model_name="openai/clip-vit-large-patch14-336",
    image_hf_dataset_id="crag-mm-2025/image-search-index-public-test",
    image_hf_dataset_tag="main",
    text_model_name="BAAI/bge-large-en-v1.5",
    web_hf_dataset_id="crag-mm-2025/web-search-index-public-test",
    web_hf_dataset_tag="v0.5",
)


## public_test with full web content
search_pipeline = UnifiedSearchPipeline(
    image_model_name="openai/clip-vit-large-patch14-336",
    image_hf_dataset_id="crag-mm-2025/image-search-index-public-test",
    text_model_name="BAAI/bge-large-en-v1.5",
    web_hf_dataset_id="crag-mm-2025/web-search-index-public-test",
    web_hf_dataset_id_private="crag-mm-2025/web-search-contents-db-private-v0.1.2",
)
# result keys: ['index', 'score', 'page_name', 'page_url', 'page_snippet', 'page_result', 'page_last_modified', 'search_result_type']
```


### Search for images

```python
# use PIL image as input (alternatively, can use image_url as input)
import requests
from PIL import Image
from io import BytesIO

from crag_image_loader import ImageLoader


image_url = "https://upload.wikimedia.org/wikipedia/commons/b/b2/The_Beekman_tower_1_%286214362763%29.jpg"
image = ImageLoader(image_url).get_image()

results = search_pipeline(image, k = 2)
assert results is not None, "No results found"
print(f"Image search results for: '{image_url}'\n")

for result in results:
    print(result)
    print('\n')
```

#### Output
```
Image search results for: 'https://upload.wikimedia.org/wikipedia/commons/b/b2/The_Beekman_tower_1_%286214362763%29.jpg'

{'index': 17030, 'score': 0.906402587890625, 'url': 'https://upload.wikimedia.org/wikipedia/commons/3/34/The_Beekman_tower_from_the_East_River_%286215420371%29.jpg', 'entities': [{'entity_name': '8 Spruce Street', 'entity_attributes': {'name': '8 Spruce Street<br />(New York by Gehry)', 'image': '8 Spruce Street (01030p).jpg', 'image_size': '200px', 'address': '8 Spruce Street<br />[[Manhattan]], New York, U.S. 10038', 'mapframe_wikidata': 'yes', 'coordinates': '{{coord|40|42|39|N|74|00|20|W|region:US-NY_type:landmark|display|=|inline,title}}', 'status': 'Complete', 'start_date': '2006', 'completion_date': '2010', 'opening': 'February 2011', 'building_type': '[[Mixed-use development|Mixed-use]]', 'architectural_style': '[[Deconstructivism]]', 'roof': '{{convert|870|ft|m|0|abbr|=|on}}', 'top_floor': '{{convert|827|ft|abbr|=|on}}', 'floor_count': '76', 'floor_area': '{{convert|1000000|sqft|m2|abbr|=|on}}', 'architect': '[[Frank Gehry]]', 'structural_engineer': '[[WSP Group|WSP Cantor Seinuk]]', 'main_contractor': 'Kreisler Borg Florman', 'developer': '[[Forest City Ratner]]', 'engineer': '[[Jaros, Baum & Bolles]] (MEP)', 'owner': '8 Spruce (NY) Owner LLC', 'management': 'Beam Living', 'website': '{{URL|https://live8spruce.com/}}'}}]}

{'index': 16196, 'score': 0.8643585443496704, 'url': 'https://upload.wikimedia.org/wikipedia/commons/1/12/New_York_by_Gehry_-_New_York_-_USA_-_panoramio.jpg', 'entities': [{'entity_name': '8 Spruce Street', 'entity_attributes': {'name': '8 Spruce Street<br />(New York by Gehry)', 'image': '8 Spruce Street (01030p).jpg', 'image_size': '200px', 'address': '8 Spruce Street<br />[[Manhattan]], New York, U.S. 10038', 'mapframe_wikidata': 'yes', 'coordinates': '{{coord|40|42|39|N|74|00|20|W|region:US-NY_type:landmark|display|=|inline,title}}', 'status': 'Complete', 'start_date': '2006', 'completion_date': '2010', 'opening': 'February 2011', 'building_type': '[[Mixed-use development|Mixed-use]]', 'architectural_style': '[[Deconstructivism]]', 'roof': '{{convert|870|ft|m|0|abbr|=|on}}', 'top_floor': '{{convert|827|ft|abbr|=|on}}', 'floor_count': '76', 'floor_area': '{{convert|1000000|sqft|m2|abbr|=|on}}', 'architect': '[[Frank Gehry]]', 'structural_engineer': '[[WSP Group|WSP Cantor Seinuk]]', 'main_contractor': 'Kreisler Borg Florman', 'developer': '[[Forest City Ratner]]', 'engineer': '[[Jaros, Baum & Bolles]] (MEP)', 'owner': '8 Spruce (NY) Owner LLC', 'management': 'Beam Living', 'website': '{{URL|https://live8spruce.com/}}'}}]}
```

### Search for text queries

```python
# Search the pipeline with a text query
text_query='What to know about Andrew Cuomo?'
results = search_pipeline(text_query, k=2)
assert results is not None, "No results found"
print(f"Web search results for: '{text_query}'\n")

for result in results:
    print(result)
    print('\n')
```

#### Output
```
Web search results for: 'What to know about Andrew Cuomo?'

{'index': 'https://en.wikipedia.org/wiki/Mario_Cuomo_chunk_2', 'score': 0.5727531909942627, 'page_name': 'Mario Cuomo - Wikipedia', 'page_snippet': 'He vigorously attacked Ronald Reagan&#x27;s ... brought him to national attention, most memorably saying: &quot;There is despair, Mr. President, in the faces that you don&#x27;t see, in the places that you don&#x27;t visit, in your shining city.&quot; He was immediately considered one of the frontrunners for the Democratic ...He vigorously attacked Ronald Reagan\'s ... brought him to national attention, most memorably saying: "There is despair, Mr. President, in the faces that you don\'t see, in the places that you don\'t visit, in your shining city." He was immediately considered one of the frontrunners for the Democratic nomination for president in 1988 and 1992. He vigorously attacked Ronald Reagan\'s record and policies in his Tale of Two Cities speech that brought him to national attention, most memorably saying: "There is despair, Mr. President, in the faces that you don\'t see, in the places that you don\'t visit, in your shining city." He was immediately considered one of the frontrunners for the Democratic nomination for president in 1988 and 1992. Cuomo was re-elected in 1986 against Republican nominee Andrew P. At its 1983 commencement ceremonies, Barnard College awarded Cuomo its highest honor, the Barnard Medal of Distinction. Also in 1983, Yeshiva University awarded him an honorary Doctor of Laws degree. In 2017, Governor Andrew Cuomo signed legislation officially naming the Tappan Zee Bridge replacement the "Governor Mario M. They had three daughters, twins Cara Ethel and Mariah Matilda Cuomo, born on January 11, 1995; and Michaela Andrea Cuomo, born on August 26, 1997. The couple divorced in 2005. Andrew served as Secretary of Housing and Urban Development under President Bill Clinton from 1997 to 2001. In his first attempt to succeed his father, he ran as Democratic candidate for New York governor in 2002, but withdrew before the primary. In November 2006, Andrew was elected New York State Attorney General; and on November 2, 2010, he was elected Governor of New York, inaugurated on January 1, 2011, and was re-elected two more times, serving until he resigned in August 2021 due to sexual harassment allegations. Cuomo\'s younger son Chris was a journalist on the ABC Network news magazine Primetime. He anchored news segments and served as co-host on Good Morning America, before moving to CNN in 2013, where he co-hosted the morning news magazine New Day.', 'page_url': 'https://en.wikipedia.org/wiki/Mario_Cuomo'}

{'index': 'https://nymag.com/intelligencer/article/why-does-andrew-cuomo-want-to-be-mayor-of-new-york-city.html_chunk_7', 'score': 0.5710090398788452, 'page_name': 'Why Does Andrew Cuomo Want to Be Mayor of New York City?', 'page_snippet': 'After being forced out as governor, he says he’s grown and learned. His brute-force takeover of the mayor’s race, at least, looks familiar, reports David Freedlander.He says he’s grown and learned. His brute-force takeover of the mayor’s race, at least, looks familiar He knew there was a rank of fractious leftists in Albany who wanted him gone but not that legislative allies would fold so quickly or that James would deliver such a devastating final report. Cuomo resigned as governor 2021, facing imminent impeachment. Photo: Andrew Kelly/Reuters · He had seen all of this unfold before, he said. And ask if they will rank him first on their ballots in the city’s ranked-choice-voting system, and the number drops to the 30s—proof, they said, that his current strength is a mirage. Other recent polling shows that this theory might not be right. A trio of late March polls put Cuomo at around 40 percent, suggesting that since entering the race, Cuomo has, if anything, increased his lead — that he was, in fact, becoming if not better known, at least better liked. The Cuomo campaign can scarcely believe their good fortune — as much as Cuomo likes to deride his opponents as far-left members of the Democratic Socialists of America, Mamdani, whose viral online videos have turned him into a phenom in the early part of the campaign, actually is a member of the group. It is the belief among several of Cuomo’s advisers that Mamdani can’t possibly break 50 percent but that he could win over the left and deprive oxygen and attention from Cuomo’s other competitors, creating a dynamic much like Cuomo faced when he ran statewide against Zephyr Teachout and Cynthia Nixon, both candidates beloved by the progressive left but with little traction beyond it.', 'page_url': 'https://nymag.com/intelligencer/article/why-does-andrew-cuomo-want-to-be-mayor-of-new-york-city.html'}
```
Can ignore the following section:
Note: The Search APIs only return urls for images and webpages, instead of full contents. To get the full webpage contents and images, you will have to download it yourself. During the challenge, participants can assume that the connection to these urls are available. 

#### Fetching the full page content

We provide a helper class to fetch the complete contents of the page. 

```python
from crag_web_result_fetcher import WebSearchResult

# Search the pipeline with a text query
text_query='What to know about Andrew Cuomo?'
results = search_pipeline(text_query, k=2)
assert results is not None, "No results found"
print(f"Web search results for: '{text_query}'\n")

for result in results:
    result = WebSearchResult(result)
    print(result["page_content"])  # prints the full page contents
```
