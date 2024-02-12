import { MemoryVectorStore } from "langchain/vectorstores/memory";
import { OpenAIEmbeddings } from "@langchain/openai";
import { HuggingFaceTransformersEmbeddings } from "@langchain/community/embeddings/hf_transformers";
import { ChatOllama } from "@langchain/community/chat_models/ollama";
import { WebPDFLoader } from "langchain/document_loaders/web/pdf";
import { RecursiveCharacterTextSplitter } from "langchain/text_splitter";
import {
    ChatPromptTemplate,
    MessagesPlaceholder,
    PromptTemplate,
} from "@langchain/core/prompts";
import { createRetrievalChain } from "langchain/chains/retrieval";
import { createStuffDocumentsChain } from "langchain/chains/combine_documents";
import { createHistoryAwareRetriever } from "langchain/chains/history_aware_retriever";
import { RunnableSequence, RunnablePick } from "@langchain/core/runnables";
import {
    AIMessage,
    HumanMessage,
} from "@langchain/core/messages";
import { ollamaURL, ollamaModel, RESPONSE_SYSTEM_TEMPLATE } from "./constants.js";
import { readPDF } from "./file.js";


function getEmbeddings(provider) {
    if (provider === "openai") {
        const openaiEmbeddings = new OpenAIEmbeddings({
            openAIApiKey: process.env.OPENAI_API_KEY, // In Node.js defaults to process.env.OPENAI_API_KEY
            batchSize: 512, // Default value if omitted is 512. Max is 2048
            modelName: "text-embedding-3-large",
        });

        return openaiEmbeddings;
    }

    const embeddingsHF = new HuggingFaceTransformersEmbeddings({
        modelName: "nomic-ai/nomic-embed-text-v1",
        // Can use "Xenova/all-MiniLM-L6-v2" for less powerful but faster embeddings
    });

    return embeddingsHF;
}

async function initializeVectorWithPDF(pdfPath) {
    const blob = await readPDF(pdfPath);
    const pdfLoader = new WebPDFLoader(blob, { parsedItemSeparator: " " });
    const docs = await pdfLoader.load();
    const splitter = new RecursiveCharacterTextSplitter({
        chunkSize: 500,
        chunkOverlap: 50,
    });
    const splitDocs = await splitter.splitDocuments(docs);

    const embeddings = getEmbeddings("openai");
    const vectorStore = await MemoryVectorStore.fromDocuments(splitDocs, embeddings).catch(console.log);
    return vectorStore;
}

function getOllamaChat() {
    const chat = new ChatOllama({
        baseUrl: ollamaURL,
        temperature: 0.3,
        model: ollamaModel,
    });

    return chat;
}

function _formatChatHistoryAsMessages(chatHistory) {
    return chatHistory.map((chatMessage) => {
        if (chatMessage.role === "human") {
            return new HumanMessage(chatMessage.content);
        } else {
            return new AIMessage(chatMessage.content);
        }
    });
}

async function queryVectorStore(messages, llm, vectorStore) {
    const text = messages[messages.length - 1].content;
    const chatHistory = await _formatChatHistoryAsMessages(messages.slice(0, -1));
    const responseChainPrompt = ChatPromptTemplate.fromMessages([
        ["system", RESPONSE_SYSTEM_TEMPLATE],
        new MessagesPlaceholder("chat_history"),
        ["user", `{input}`],
    ]);
    const documentChain = await createStuffDocumentsChain({
        llm,
        prompt: responseChainPrompt,
        documentPrompt: PromptTemplate.fromTemplate(
            `<doc>\n{page_content}</doc>`
        ),
    });
    const historyAwarePrompt = ChatPromptTemplate.fromMessages([
        new MessagesPlaceholder("chat_history"),
        ["user", `{input}`],
        [
            "user",
            "Given the above conversation, generate a natural language search query to look up in order to get information relevant to the conversation. Do not respond with anything except the query."
        ],
    ]);
    const historyAwareRetrieverChain = await createHistoryAwareRetriever({
        llm,
        retriever: vectorStore.asRetriever(),
        rephrasePrompt: historyAwarePrompt,
    });
    const retrievalChain = await createRetrievalChain({
        combineDocsChain: documentChain,
        retriever: historyAwareRetrieverChain,
    });
    const fullChain = RunnableSequence.from([
        retrievalChain,
        new RunnablePick("answer"),
    ]);
    const stream = await fullChain.stream({
        input: text,
        chat_history: chatHistory,
    });

    let answer = "";

    for await (const chunk of stream) {
        if (chunk) {
            answer += chunk;
        }
    }

    return answer;
}

export default {
    ollama: true,
    initializeVectorWithPDF,
    getOllamaChat,
    queryVectorStore,
};
