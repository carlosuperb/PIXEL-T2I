// webapp/frontend/prompt_generator.js
//
// Random semantic character description generator.
//
// This module produces a concise, comma-separated natural language prompt that
// describes a character's body type, clothing, hair, and optional gear.
//
// Notes:
// - Style/layout anchors are intentionally excluded and are injected during inference.
// - The exported API is window.getRandomPrompt(), used by the frontend controller.

(() => {
  "use strict";

  // ---------------------------------------------------------------------------
  // Helpers
  // ---------------------------------------------------------------------------
  function pick(arr) {
    return arr[Math.floor(Math.random() * arr.length)];
  }

  function maybe(p) {
    return Math.random() < p;
  }

  function joinWithAnd(items) {
    if (!items || items.length === 0) return "";
    if (items.length === 1) return items[0];
    if (items.length === 2) return `${items[0]} and ${items[1]}`;
    return `${items.slice(0, -1).join(", ")} and ${items[items.length - 1]}`;
  }

  function uniqueKeepOrder(items) {
    const seen = new Set();
    const out = [];
    for (const x of items) {
      if (!x) continue;
      if (seen.has(x)) continue;
      seen.add(x);
      out.push(x);
    }
    return out;
  }

  // ---------------------------------------------------------------------------
  // Token banks (semantic-only)
  // ---------------------------------------------------------------------------
  const GENDERS = ["male", "female"];
  const RACES = ["human", "dark-elf", "orc", "skeleton"];

  const HUMAN_SKIN = ["light skin", "tanned skin", "dark skin"];

  const HAIR_COLORS = [
    "black",
    "blonde",
    "blue",
    "brown",
    "gray",
    "green",
    "pink",
    "purple",
    "white",
    "dark blonde",
    "light blonde",
    "white blonde",
    "ruby red",
  ];

  const HAIR_STYLE_LABELS = [
    "princess hairstyle",
    "pageboy hairstyle",
    "unkempt hairstyle",
    "messy hairstyle",
    "bedhead hairstyle",
    "swoop hairstyle",
    "long mohawk hairstyle",
    "short mohawk hairstyle",
    "mohawk hairstyle",
    "long hairstyle",
    "extra-long hairstyle",
    "long topknot hairstyle",
    "short topknot hairstyle",
    "topknot hairstyle",
    "pixie cut hairstyle",
    "ponytail hairstyle",
    "short hairstyle",
    "parted hairstyle",
    "plain hairstyle",
  ];

  const TORSO_ITEMS = [
    "tunic",
    "robe",
    "shirt",
    "long-sleeve shirt",
    "sleeveless shirt",
    "pirate shirt",
    "tabard jacket",
    "cape",
    "tattered cape",
    "trimmed cape",
    "vest",
  ];

  const LEGS_ITEMS = ["pants", "skirt", "robe skirt", "greaves"];
  const HEADGEAR_ITEMS = ["cap", "helm", "hood", "bandana", "tiara"];
  const HAND_ITEMS = ["gloves", "bracers", "bracelets"];
  const FEET_ITEMS = ["boots", "shoes", "slippers", "ghillies"];

  const MATERIALS = ["gold", "silver", "bronze", "iron", "steel", "metal", "chain", "leather", "cloth", "wood"];
  const COLORS = [
    "black",
    "white",
    "gray",
    "brown",
    "maroon",
    "lavender",
    "green",
    "blue",
    "purple",
    "red",
    "magenta",
    "pink",
    "yellow",
    "teal",
    "cyan",
  ];

  const WEAPONS = ["dagger", "spear", "shield"];

  // ---------------------------------------------------------------------------
  // Phrase builders
  // ---------------------------------------------------------------------------
  /**
   * Produce a short item phrase by optionally prefixing with a material or color.
   *
   * @param {string} baseType - Item type label (e.g., "boots", "cape").
   * @param {boolean} [preferMaterial=true] - Whether material prefixing is preferred.
   * @returns {string} Rendered phrase.
   */
  function materialOrColorPhrase(baseType, preferMaterial = true) {
    if (preferMaterial && maybe(0.7)) {
      const m = pick(MATERIALS);
      return `${m} ${baseType}`;
    }
    if (maybe(0.8)) {
      const c = pick(COLORS);
      return `${c} ${baseType}`;
    }
    return baseType;
  }

  /**
   * Build a body description phrase, including race-specific or human skin tone.
   *
   * @returns {string} Body phrase (e.g., "female human body, tanned skin").
   */
  function buildBodyPhrase() {
    const gender = pick(GENDERS);
    const race = pick(RACES);

    const body = `${gender} ${race} body`;
    let skin = null;

    if (race === "orc") skin = "green skin";
    else if (race === "dark-elf") skin = "purple skin";
    else if (race === "human") skin = pick(HUMAN_SKIN);

    return skin ? `${body}, ${skin}` : body;
  }

  /**
   * Build an optional hair description phrase.
   *
   * @returns {string|null} Hair phrase or null when omitted.
   */
  function buildHairPhrase() {
    if (!maybe(0.9)) return null;
    const color = pick(HAIR_COLORS);
    const style = pick(HAIR_STYLE_LABELS);
    return `${color} ${style}`;
  }

  /**
   * Build a list of clothing phrases with simple diversity and deduplication.
   *
   * @returns {string[]} Clothing phrases.
   */
  function buildWearingPhrases() {
    const clothing = [];

    // Torso and footwear are always included to avoid overly sparse prompts.
    clothing.push(materialOrColorPhrase(pick(TORSO_ITEMS)));
    if (maybe(0.9)) clothing.push(materialOrColorPhrase(pick(LEGS_ITEMS)));
    if (maybe(0.45)) clothing.push(materialOrColorPhrase(pick(HEADGEAR_ITEMS)));
    if (maybe(0.4)) clothing.push(materialOrColorPhrase(pick(HAND_ITEMS)));
    clothing.push(materialOrColorPhrase(pick(FEET_ITEMS)));

    return uniqueKeepOrder(clothing);
  }

  /**
   * Build a list of optional gear phrases.
   *
   * @returns {string[]} Gear phrases.
   */
  function buildGearPhrases() {
    const gear = [];

    // Include gear with a moderate probability to keep prompts varied.
    if (maybe(0.45)) {
      const w = pick(WEAPONS);
      gear.push(w);
    }

    return uniqueKeepOrder(gear);
  }

  // ---------------------------------------------------------------------------
  // Main generator
  // ---------------------------------------------------------------------------
  /**
   * Build a full prompt caption as a comma-separated list of segments.
   *
   * Output format:
   * - "<gender> <race> body, <skin>, wearing <items>, with <hair>, with <gear>"
   *
   * @returns {string} Prompt caption.
   */
  function buildCaption() {
    const segments = [];

    const body = buildBodyPhrase();
    if (body) segments.push(body);

    const wearing = buildWearingPhrases();
    if (wearing.length > 0) segments.push(`wearing ${joinWithAnd(wearing)}`);

    const hair = buildHairPhrase();
    if (hair) segments.push(`with ${hair}`);

    const gear = buildGearPhrases();
    if (gear.length > 0) segments.push(`with ${joinWithAnd(gear)}`);

    return segments.join(", ");
  }

  // ---------------------------------------------------------------------------
  // Public API
  // ---------------------------------------------------------------------------
  /**
   * Generate a random semantic character prompt for text-conditioned inference.
   *
   * @returns {string} Random prompt string.
   */
  window.getRandomPrompt = function getRandomPrompt() {
    return buildCaption();
  };
})();
