import { pipeline } from "@xenova/transformers";
import fs from "fs";
import dotenv from "dotenv";
import util from "util";
dotenv.config();

const text = fs.readFileSync("./documents/test.txt", "utf-8");
const answerer = await pipeline("question-answering");
const question = "What is the password?";
const context = text;

try {
    const result = await answerer(question, context);
    console.log(util.inspect(result, false, null, true));
} catch (error) {
    console.error(error);
}
