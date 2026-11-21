import { z } from "zod";
import { createDecipheriv, createHmac, hkdfSync, timingSafeEqual } from "node:crypto";
import type { AppInstance } from "~~/server/utils/app-db";
import type { MessageRecord } from "~~/server/utils/evolution-db";

/**
 * Forward one stored WhatsApp message to another chat, its media included.
 *
 * The relay problem this exists for: a model that can read a chat but only send
 * text can forward the *words* of an image and never the image. Wiring the
 * bytes through the model is not the answer either — an MCP response carrying
 * base64 photo bytes fights the 2 MiB request cap of the route (see
 * `MCP_BODY_LIMIT` in body-limit.ts) at both ends and lands in the context
 * window besides. So this tool fetches, decrypts and re-sends server-side, and
 * the model only ever passes ids.
 *
 * **Why re-send rather than a native forward.** Evolution (and Baileys under
 * it) has no server-side "forward" that preserves provenance: WhatsApp for
 * Business APIs never expose one. Re-sending via `/message/sendMedia` is the
 * supported path everywhere; it arrives as a new message from this account,
 * which is exactly what a relay wants — the human operator does the same thing
 * by hand today.
 *
 * **Where the bytes come from, in order of preference:**
 *
 *   1. **Fresh decrypt from WhatsApp.** A Baileys media payload always carries
 *      the encrypted file's own download URL and the `mediaKey` the recipient
 *      derives the AES key material from (HKDF-SHA256, then AES-256-CBC with a
 *      trailing HMAC-SHA256 check). The algorithm is Baileys' own
 *      `downloadMediaMessage` inlined here — implemented, not vendored, because
 *      the web app must not grow a dependency on Baileys for one tool, and
 *      because it is ~40 lines.
 *   2. **Evolution's copy.** Reforged ≥ 3, with S3 enabled, downloads incoming
 *      media once, stores it and patches the row's content object with
 *      `mediaUrl` (see `whatsapp.baileys.service.ts` there: `messageRaw.message
 *      .= mediaUrl`). That copy does not expire as fast as WhatsApp's own URLs.
 *      Used as fallback when the fresh decrypt fails.
 *
 * **Caption and the token's disclaimer.** `send-text-message` appends the
 * token's disclaimer because the text there is *the model's own words*; a
 * forward's text is somebody else's, relayed 1:1, and appending the owner's
 * sign-off would put words in the relayed party's caption. Nothing is appended;
 * `sendDisclaimer` is deliberately not read here.
 *
 * **Security boundary: the source chat is scopes-checked, not trusted.** The
 * `sourceJid` goes through `expandChat` — the same literal-scope check
 * `read-messages` uses, including the LID/phone-number link widening an
 * in-scope token gets — and only the in-scope JIDs it returns flow into the
 * SQL predicate. A token cannot forward what it cannot read. The target goes
 * through the send guard chain of `send-text-message` unchanged:
 * `assertNotBareLid` first, then `assertNumberAllowed`, which fails closed.
 *
 * **Bytes never reach the MCP response.** The tool answers with ids and sizes;
 * the payload travels server-to-server to Evolution. That also means nothing
 * here needs a body-limit exception, and a media file cannot be smuggled out
 * through the read surface this tool offers.
 */
export default defineKindTool({
  name: "forward-message",
  kind: "whatsapp",
  title: "Forward WhatsApp message",
  description:
    "Forward one stored message — text, image, video, audio, document or sticker — " +
    "from its chat to another chat, media and caption included. Pass the " +
    "`messageId` exactly as read-messages or search-messages returned it for the " +
    "message, and full JIDs for both `sourceJid` and `targetJid`, exactly as those " +
    "tools return them. Both chats must be in this token's allowed list: the source " +
    "is checked like a read, the target like a send. The message is re-sent from the " +
    "connected account — WhatsApp has no server-side forward, so it arrives as a new " +
    "message from that account, content unchanged: a caption is forwarded verbatim and " +
    "the token's send disclaimer is not appended to it. The account must be in the " +
    "`open` state — check with get-connection-status first. A @-mention inside the " +
    "forwarded text travels as plain text, not as a live mention. Only recent history " +
    "is scanned for the id: when the tool cannot find the message, re-read the chat to " +
    "confirm the id and recency before retrying. Cannot be undone once delivered.",
  annotations: {
    readOnlyHint: false,
    destructiveHint: false,
    idempotentHint: false,
    openWorldHint: true,
  },
  // Media forwarding decrypts a file and uploads it again: nominate it for the
  // separate, lower per-connection budget for costly tools (mcp-rate-limit.ts).
  expensive: true,
  inputSchema: {
    sourceJid: z
      .string()
      .min(1)
      .describe(
        "The chat the message lives in, exactly as list-chats or read-messages return it " +
          "(e.g. 79972425314508@lid or 5511999999999@s.whatsapp.net)",
      ),
    messageId: z.string().min(1).describe("The message's id, exactly as read-messages or search-messages returned it"),
    targetJid: z
      .string()
      .min(1)
      .describe(
        "The chat to send it to, exactly as list-chats or read-messages return it " +
          "(e.g. 120363414123201385@g.us). A @lid is a privacy alias: never send to its digits alone",
      ),
  },
  handler: async ({ sourceJid, messageId, targetJid }) => {
    const { instance, scope } = useMcpAuth();

    // ---- Target: the send guard chain of send-text-message, unchanged. ----
    // LID first, so stripped digits are refused as that rather than resolved
    // as a phone number by the scope check.
    await assertNotBareLid(instance, targetJid);
    // Resolves through Evolution and compares exact JIDs; fails closed.
    await assertNumberAllowed(instance, scope, targetJid);

    // ---- Source: the read scope of read-messages, unchanged. ----
    // Throws 403 when the source chat is not in this token's allowed list;
    // returns only the in-scope JIDs, and exactly those flow into the query.
    const links = await lidLinks(instance);
    const chat = expandChat(scope, links, sourceJid);

    const payload = await messagePayloadById(instance, chat.jids, messageId);
    if (!payload) {
      throw createError({
        statusCode: 404,
        message:
          `No readable message with id "${messageId}" in this chat's recent history. ` +
          "Re-read the chat with read-messages to confirm the id — it must be the one that tool returned, " +
          "and the message must be within the newest few thousand.",
      });
    }

    const content = contentOf(payload);

    // ---- Text messages: same endpoint send-text-message uses. ----
    const text = textOf(content);
    if (text !== undefined) {
      const evolution = useEvolutionClient();
      let result: Record<string, unknown>;
      try {
        result = await evolution<Record<string, unknown>>(`/message/sendText/${encodeURIComponent(instance.name)}`, {
          method: "POST",
          body: { number: targetJid, text },
        });
      } catch (error) {
        relayEvolutionSendError(error, targetJid, instance.id);
      }
      noteSocketAlive(instance);
      return { forwarded: "text", messageId, from: sourceJid, to: targetJid, ...result };
    }

    // ---- Media messages: decrypt server-side, re-send via sendMedia. ----
    const media = mediaShapeOf(content);
    if (!media) {
      const type = outerTypeOf(content) || "an unrecognised record";
      throw createError({
        statusCode: 422,
        message:
          `This message is ${type}, which forward-message cannot relay. It forwards ` +
          "plain text, images, video, audio, documents and stickers.",
      });
    }

    const raw = content[media.key] as Record<string, any>;
    const bytes = await forwardableBytes(content, raw, media);
    const mimetype = typeof raw.mimetype === "string" && raw.mimetype ? raw.mimetype : media.fallbackMimetype;
    const caption = typeof raw.caption === "string" ? raw.caption : undefined;

    const evolution = useEvolutionClient();
    let result: Record<string, unknown>;
    try {
      result = await evolution<Record<string, unknown>>(`/message/sendMedia/${encodeURIComponent(instance.name)}`, {
        method: "POST",
        body: {
          number: targetJid,
          mediatype: media.mediatype,
          mimetype,
          media: bytes.toString("base64"),
          ...(typeof raw.fileName === "string" && raw.fileName && { fileName: raw.fileName }),
          ...(caption && { caption }),
        },
      });
    } catch (error) {
      relayEvolutionSendError(error, targetJid, instance.id);
    }
    noteSocketAlive(instance);
    return {
      forwarded: media.mediatype,
      messageId,
      from: sourceJid,
      to: targetJid,
      ...(caption !== undefined && { captionForwarded: caption }),
      bytes: bytes.length,
      mimetype,
      ...result,
    };
  },
});

/* ------------------------------------------------------------------------- *
 * Row access
 * ------------------------------------------------------------------------- */

/**
 * How far back to look for the id. A relay forwards traffic that just
 * arrived, so a scan over the newest few thousand covers every honest caller;
 * a bound is what keeps a wrong id from becoming a table scan billed to the
 * tool. `listMessagesPage` already deduplicates rows and scopes by JID in SQL,
 * so each page here is one page of exactly the chats this token may read.
 */
const SCAN_PAGE_SIZE = 200;
const SCAN_MAX_PAGES = 15;

/**
 * The stored content object for one message id, or undefined when it is not in
 * the newest `SCAN_PAGE_SIZE * SCAN_MAX_PAGES` messages of these chats.
 */
async function messagePayloadById(instance: AppInstance, jids: readonly string[], messageId: string): Promise<unknown> {
  for (let page = 1; page <= SCAN_MAX_PAGES; page++) {
    const { records } = await listMessagesPage(instance, { jids, limit: SCAN_PAGE_SIZE, page });
    const hit = records.find((record: MessageRecord) => record.key?.id === messageId);
    if (hit) return hit.message;
    if (!records.length) break;
  }
  return undefined;
}

/* ------------------------------------------------------------------------- *
 * Payload classification
 * ------------------------------------------------------------------------- */

/**
 * Transparent FutureProofMessage wrappers, the same family chats.ts unwraps:
 * album media and caption edits reach the row nested. Bounded, as there, since
 * the payload is JSON-derived and only absurd nesting needs capping.
 */
const WRAPPER_KEYS = ["associatedChildMessage", "documentWithCaptionMessage", "editedMessage"] as const;
const MAX_UNWRAP_DEPTH = 4;

function contentOf(payload: unknown): Record<string, any> {
  let current: any = payload;
  for (let depth = 0; depth < MAX_UNWRAP_DEPTH && current && typeof current === "object"; depth++) {
    if (typeof current !== "object") return current;
    const wrapper = WRAPPER_KEYS.find((key) => current[key] !== undefined);
    if (!wrapper) return current as Record<string, any>;
    const inner = current[wrapper];
    // `associatedChildMessage`/`documentWithCaptionMessage` wrap
    // `{ message: … }`; `editedMessage` *is* the content.
    current = inner && typeof inner === "object" && "message" in inner ? inner.message : inner;
  }
  return (current ?? {}) as Record<string, any>;
}

/** The body of a forwardable text message, or undefined when this is not one. */
function textOf(content: Record<string, any>): string | undefined {
  if (typeof content.conversation === "string" && content.conversation !== "") return content.conversation;
  const extended = content.extendedTextMessage;
  if (extended && typeof extended.text === "string" && extended.text !== "") return extended.text;
  return undefined;
}

/** What Evolution's sendMedia route calls the content we found. */
interface MediaShape {
  key: string;
  mediatype: "image" | "video" | "audio" | "document";
  /** HKDF info string — Baileys' "WhatsApp … Keys" per content type. */
  keyInfo: string;
  fallbackMimetype: string;
}

/**
 * Stickers forward as images: their payload is WebP media with image-derived
 * keys, and `sendMedia` has no sticker type. `mediatype` values are the enum in
 * Evolution's mediaMessageSchema, re-verified on a tag bump.
 */
const MEDIA_SHAPES: Record<string, MediaShape> = {
  imageMessage: {
    key: "imageMessage",
    mediatype: "image",
    keyInfo: "WhatsApp Image Keys",
    fallbackMimetype: "image/jpeg",
  },
  stickerMessage: {
    key: "stickerMessage",
    mediatype: "image",
    keyInfo: "WhatsApp Image Keys",
    fallbackMimetype: "image/webp",
  },
  videoMessage: {
    key: "videoMessage",
    mediatype: "video",
    keyInfo: "WhatsApp Video Keys",
    fallbackMimetype: "video/mp4",
  },
  audioMessage: {
    key: "audioMessage",
    mediatype: "audio",
    keyInfo: "WhatsApp Audio Keys",
    fallbackMimetype: "audio/ogg",
  },
  documentMessage: {
    key: "documentMessage",
    mediatype: "document",
    keyInfo: "WhatsApp Document Keys",
    fallbackMimetype: "application/octet-stream",
  },
};

function mediaShapeOf(content: Record<string, any>): MediaShape | undefined {
  for (const shape of Object.values(MEDIA_SHAPES)) {
    const raw = content[shape.key];
    if (raw && typeof raw === "object") return shape;
  }
  return undefined;
}

/** The outermost content key, to name what was refused. */
function outerTypeOf(content: Record<string, any>): string | undefined {
  if (!content || typeof content !== "object") return undefined;
  const key = Object.keys(content).find((k) => k.endsWith("Message"));
  return key ? `a ${key}` : undefined;
}

/* ------------------------------------------------------------------------- *
 * Media bytes
 * ------------------------------------------------------------------------- */

/** Hard ceiling on a forwarded file: base64 through the app process, not out to the caller. */
const MAX_FORWARD_BYTES = 16 * 1024 * 1024;

/**
 * The bytes to re-send, trying the fresh WhatsApp decrypt first and
 * Evolution's own copy (its S3 `mediaUrl`, patched into the content object by
 * reforged when storage is enabled) second. WhatsApp media URLs age out, so
 * for an old message the second path is the one that still works; for a fresh
 * one the first path needs no storage configured at all.
 */
async function forwardableBytes(
  content: Record<string, any>,
  raw: Record<string, any>,
  media: MediaShape,
): Promise<Buffer> {
  const errors: string[] = [];

  // --- 1. Fresh decrypt ---
  if (typeof raw.url === "string" && typeof raw.mediaKey === "string") {
    try {
      return await decryptedFromWhatsApp(raw, media.keyInfo);
    } catch (error) {
      errors.push(error instanceof Error ? error.message : String(error));
    }
  }

  // --- 2. Evolution's copy ---
  const stored =
    typeof content.mediaUrl === "string"
      ? content.mediaUrl
      : typeof raw.mediaUrl === "string"
        ? raw.mediaUrl
        : undefined;
  if (stored) {
    try {
      const bytes = await download(stored);
      if (bytes.length > 0 && bytes.length <= MAX_FORWARD_BYTES) return bytes;
      errors.push(`Evolution's copy gave invalid bytes (${bytes.length})`);
    } catch (error) {
      errors.push(`Evolution's stored copy failed: ${error instanceof Error ? error.message : String(error)}`);
    }
  }

  throw createError({
    statusCode: 502,
    message:
      "The media could not be fetched, so nothing was sent. Tried: " +
      (errors.length ? errors.join("; ") : "no usable source URL was stored on the message. ") +
      "Download URLs of very old messages age out; when Evolution storage (S3) is enabled, newer messages keep a copy.",
  });
}

/** `https:` ONLY: the URL arrives in message data, so constrain what fetch will touch. */
async function download(url: string): Promise<Buffer> {
  if (!/^https:\/\//.test(url)) throw new Error(`refused a non-https media URL (${new URL(url).protocol})`);
  const bytes = await $fetch(url, { responseType: "arrayBuffer", retry: 0 });
  const buffer = Buffer.from(bytes as ArrayBuffer);
  if (buffer.length > MAX_FORWARD_BYTES) {
    throw new Error(`the file is ${buffer.length} bytes, over the ${MAX_FORWARD_BYTES} this tool forwards`);
  }
  return buffer;
}

/**
 * Baileys' media decryption, done here so the web app never depends on Baileys
 * for one tool: HKDF-SHA256(mediaKey, 112 bytes) → IV ‖ cipherKey ‖ macKey,
 * AES-256-CBC over the downloaded ciphertext, HMAC-SHA256 over all but the
 * last 10 bytes checked against those last 10, then trim to `fileLength`.
 * Key material stays in this function's frame.
 */
async function decryptedFromWhatsApp(raw: Record<string, any>, keyInfo: string): Promise<Buffer> {
  const encrypted = await download(raw.url as string);

  const expanded = Buffer.from(
    hkdfSync("sha256", Buffer.from(raw.mediaKey as string, "base64"), Buffer.alloc(0), Buffer.from(keyInfo), 112),
  );
  const iv = expanded.subarray(0, 16);
  const cipherKey = expanded.subarray(16, 48);
  const macKey = expanded.subarray(48, 80);

  if (encrypted.length < 16 + 10) throw new Error("the encrypted media is too short to be media");

  const ciphertext = encrypted.subarray(0, encrypted.length - 10);
  const messageMac = encrypted.subarray(encrypted.length - 10);
  const mac = createHmac("sha256", macKey).update(iv).update(ciphertext).digest();
  if (!timingSafeEqual(mac.subarray(0, 10), messageMac)) {
    throw new Error("the media failed its integrity check (HMAC mismatch)");
  }

  const decipher = createDecipheriv("aes-256-cbc", cipherKey, iv);
  const plain = Buffer.concat([decipher.update(ciphertext), decipher.final()]);

  const length = lengthOf(raw.fileLength);
  const bytes = length !== undefined && length <= plain.length ? plain.subarray(0, length) : plain;
  if (bytes.length === 0) throw new Error("decryption produced no bytes");
  if (bytes.length > MAX_FORWARD_BYTES) {
    throw new Error(`the decrypted file is over the ${MAX_FORWARD_BYTES} this tool forwards`);
  }
  return bytes;
}

/** Baileys serialises Long as {low, high}; older payloads as digits. */
function lengthOf(fileLength: unknown): number | undefined {
  if (typeof fileLength === "string" && /^\d+$/.test(fileLength)) return Number(fileLength);
  if (typeof fileLength === "number" && Number.isSafeInteger(fileLength)) return fileLength;
  if (fileLength && typeof fileLength === "object") {
    const { low, high } = fileLength as { low?: number; high?: number };
    if (typeof low === "number") return (typeof high === "number" ? high : 0) * 0x1_0000_0000 + (low >>> 0);
  }
  return undefined;
}
