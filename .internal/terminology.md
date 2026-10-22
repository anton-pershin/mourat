## Terminology

### General

| Term | Definition |
|------|------------|
| Content item | A paper or an Internet post collected by mourat. |
| Research question | A question that the user wants to answer through research. |
| Technical challenge | A technical problem for which the user seeks a solution. |
| Research topic | A subject of research that can be linked to research questions or technical challenges. |
| Constraint | A condition that refines the user's research questions, technical challenges, or research topics. |

### Collection pipelines

#### Shared terms

| Term | Definition |
|------|------------|
| Relevance | The relation of a content item to a specified research question, technical challenge, research topic, or constraint. |
| Relevance score | An LLM assessment from 0 to 100 of relevance to one specified research question, technical challenge, research topic, or constraint. |
| Filtering score | The highest relevance score among the scores selected for filtering. Whether constraint scores contribute depends on the collection workflow configuration. |
| Influence score | An assessment from 0 to 100 of a content item's impact or authority, using the method configured for its collection workflow. |
| Reference level | A configured raw post score used to normalize influence. Reddit uses a separate reference for each subreddit; Hacker News uses one global reference. |
| Paper candidate | A paper proposed by a collector or discovery step, before conversion to the resolved-paper representation. |
| Resolved paper | A paper in the common representation used for relevance scoring and output, with metadata obtained from a provider or converted from an arXiv candidate. |
| Provenance | The names of the discovery agent or generators that produced a paper candidate, retained with the paper. |

#### Newest papers — collect_newest_papers.py

| Term | Definition |
|------|------------|
| New submission | An arXiv RSS entry announced as new, rather than as a replacement or cross-listing. |
| Relevance triage | A preliminary boolean assessment of whether a paper is plausibly relevant to at least one specified research question, technical challenge, or research topic. Constraints are excluded. |
| Author affiliation | An institution associated with an author, extracted from the paper's arXiv HTML page. |
| Authority influence | An LLM assessment of author and institution authority, recorded as the paper's influence score. |

#### Influential papers from scratch — collect_influential_papers_from_scratch.py

| Term | Definition |
|------|------------|
| Paper discovery | An LLM agent's web search for paper candidates related to specified research questions, technical challenges, research topics, or constraints. |
| Paper resolution | Matching a candidate to a provider record to obtain paper identifiers and metadata. Candidates that cannot be resolved are removed. |
| Field-weighted citation impact (FWCI) | A provider's citation-impact measure normalized for field and publication age, preferred when available for paper influence assessment. |
| Citations per year | A citation rate derived from citation count and publication age, used when FWCI is unavailable or malformed. |
| Verified arXiv PDF | An arXiv PDF URL checked by the verifier. If no URL can be verified, the paper retains an empty URL and a reason. |

#### Influential papers from seeds — collect_influential_papers_from_seeds.py

This pipeline also uses the paper-resolution, influence-measure, and PDF terms defined for collection from scratch.

| Term | Definition |
|------|------------|
| Seed item | A stored content item selected as a starting point for paper discovery. |
| Seed resolution | Matching a seed item to a paper record with an arXiv identifier or DOI for expansion. |
| Forward citation expansion | Discovery of papers that cite a seed paper. |
| Backward reference expansion | Discovery of papers cited by a seed paper. |
| Relevance search | A provider search using a seed paper's title to find related papers. |
| Seed influence floor | A minimum candidate influence score derived from a configured percentile of the resolved seed papers' normalized influence scores. |

#### Reddit posts — collect_posts.py

| Term | Definition |
|------|------------|
| Slop | A post classified as carrying no substantive information. |
| Heuristic slop filter | A deterministic filter that removes posts using configured rules before LLM classification. |
| Slop verdict | An LLM classification of a post as slop or not slop, with a justification. |
| Additional context | Short, self-contained facts obtained from the web that are not already evident in the post. |
| Post enrichment | Adding additional context to a Reddit post before relevance scoring. |

#### Hacker News posts — collect_hn_posts.py

| Term | Definition |
|------|------------|
| Story | A Hacker News submission collected through the Algolia API, with its title, URL, points, and comment count. |
| Linked content | The main text extracted from the web page referenced by a story's URL. |
| Summary | A compact LLM description of linked content. Relevance scoring uses the title and summary, not the full fetched text. |
