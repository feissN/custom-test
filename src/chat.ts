import fs from "fs";
import { HuggingFaceInference } from "langchain/llms/hf";
import { ConversationalRetrievalQAChain, RetrievalQAChain } from "langchain/chains";
import { HuggingFaceInferenceEmbeddings } from "langchain/embeddings/hf";
import { FaissStore } from "langchain/vectorstores/faiss";
import { ConversationSummaryMemory } from "langchain/memory";
import dotenv from "dotenv";
dotenv.config();

import util from "util";
import { codeBlock, oneLine } from "common-tags";

const dbPath = process.env.DB_DIR || "./db";

const embeddingsModelName = "sentence-transformers/all-MiniLM-L6-v2";
// const modelName = "bigscience/bloom-560m";
const modelName = "gpt2";

const makeChain = (vectorstore: FaissStore) => {
    // const CONDENSE_PROMPT = `Given the following conversation and a follow up question, rephrase the follow up question to be a standalone question.

    // Chat History:
    // {chat_history}
    // Follow Up Input: {question}
    // Standalone question:`;

    // const QA_PROMPT = `You are a helpful AI assistant. Use the following pieces of context to answer the question at the end.
    // If you don't know the answer, just say you don't know. DO NOT try to make up an answer.
    // If the question is not related to the context, politely respond that you are tuned to only answer questions that are related to the context.

    // {context}

    // Question: {question}
    // Helpful answer in markdown:`;

    const QA_PROMPT = codeBlock`
        ${oneLine`
            You are a very enthusiastic Personal AI who loves
            to help people! Given the following document, answer the user's question using
            only that information, outputted in markdown format.
        `}

        ${oneLine`
            If you are unsure
            and the answer is not explicitly written in the documents, say
            "Sorry, I don't know how to help with that."
        `}

        ${oneLine`
            Always include related code snippets if available.
        `}

        ${oneLine`
            Here is the document:
            {context}
        `}

        ${oneLine`
          Answer my next question using only the above documentation.
          You must also follow the below rules when answering:
        `}
        ${oneLine`
          - Do not make up answers that are not provided in the documentation.
        `}
        ${oneLine`
          - If you are unsure and the answer is not explicitly written
          in the documentation context, say
          "Sorry, I don't know how to help with that."
        `}
        ${oneLine`
          - Prefer splitting your response into multiple paragraphs.
        `}
        ${oneLine`
          - Output as markdown with code snippets if available.
        `}

        ${oneLine`
            Here is my question:
            {question}
        `}
      `;

    // const chain = ConversationalRetrievalQAChain.fromLLM(model, vectorstore.asRetriever(), {
    //     qaTemplate: QA_PROMPT,
    //     questionGeneratorTemplate: QA_PROMPT,
    //     returnSourceDocuments: true, //The number of source documents returned is 4 by default
    //     verbose: true,
    // });
    // return chain;
};

const chat = async () => {
    console.log(`Check if vectorstore exists in '${dbPath}'`);
    if (!fs.existsSync(dbPath)) {
        throw "Vector store does not exist. Try calling api/ingest first";
    }
    console.log(`Loading vectorstore from '${dbPath}'`);
    const vectorStore = await FaissStore.load(
        dbPath,
        new HuggingFaceInferenceEmbeddings({
            model: embeddingsModelName,
        })
    );
    console.log("Vectorstore loaded");

    const model = new HuggingFaceInference({
        apiKey: process.env.HUGGINGFACEHUB_API_KEY,
        model: "TheBloke/WizardLM-13B-V1-1-SuperHOT-8K-GPTQ",
    });
    const chain = ConversationalRetrievalQAChain.fromLLM(model, vectorStore.asRetriever());

    console.log("Requesting...");
    const question = "What are documentloaders?";
    const res = await chain.call({ question, chat_history: "" });
    console.log(res);
};

(async () => {
    const start = new Date();
    await chat();
    console.log(`Duration: ${new Date().getTime() - start.getTime()}ms`);
})();
