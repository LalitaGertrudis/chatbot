import { readFile } from "fs/promises";
import { Blob } from "buffer";

export async function readPDF(filePath) {
    const file = await readFile(filePath);
    const blob = new Blob([file], { type: "application/pdf" });
    return blob;
}
