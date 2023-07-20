import dotenv from "dotenv";
import { DirectoryLoader } from "langchain/document_loaders/fs/directory";
import { PDFLoader } from "langchain/document_loaders/fs/pdf";
import { TextLoader } from "langchain/document_loaders/fs/text";
import { HuggingFaceInferenceEmbeddings } from "langchain/embeddings/hf";
import { RecursiveCharacterTextSplitter } from "langchain/text_splitter";
import { FaissStore } from "langchain/vectorstores/faiss";
dotenv.config();

const embeddingsModelName = "sentence-transformers/all-MiniLM-L6-v2";

const directory = process.env.DOCS_DIR || "./documents";
const dbPath = process.env.DB_DIR || "./db";

const ingest = async () => {
    const directoryLoader = new DirectoryLoader(directory, {
        ".pdf": (path) => new PDFLoader(path),
        ".txt": (path) => new TextLoader(path),
    });
    console.log(`Loading documents from '${directory}'`);
    const rawDocs = await directoryLoader.load();
    const textSplitter = new RecursiveCharacterTextSplitter({
        chunkSize: 1000,
        chunkOverlap: 150,
    });
    console.log(`Splitting documents in chunks`);
    const docs = await textSplitter.splitDocuments(rawDocs);

    const embeddings = new HuggingFaceInferenceEmbeddings({
        model: embeddingsModelName,
    });
    console.log(`Creating store from documents`);
    const vectorStore = await FaissStore.fromDocuments(docs, embeddings);

    console.log(`Saving store to '${dbPath}'`);
    await vectorStore.save(dbPath);
    console.log("Ingest compelte!");
};

(async () => {
    const start = new Date();
    await ingest();
    console.log(`Duration: ${new Date().getTime() - start.getTime()}ms`);
})();
