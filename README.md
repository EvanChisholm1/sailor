# SAILOR

Sailor stands for:

-   S emantic
-   A rtificial
-   I ntelligence
-   L ink
-   O rganizer
-   R etriever

This is sort of a spiritual successor to [WTFDIST](https://github.com/EvanChisholm1/wtfdist) my personal search engine. Sailor uses modern semantic search techniques to hopefully achieve much better search performance. Initially I will probably do standard KNN with numpy.array in order to find documents and then I will handroll my own ANN algorithims for funsies to make it go faster.

## todo

-   [ ] better embedding model
-   [ ] better chunking
-   [ ] better vector db/search algorithim. [Twisty?](https://github.com/EvanChisholm1/twisty)
-   [ ] rag llm chatbot
-   [ ] switch away from json file? storing numbers as text isn't exactly the smartest way to do it.
-   [ ] Avoid duplication, check if url already exists, if it does then don't add, only update if over a certain age.
