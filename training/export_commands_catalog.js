const fs = require("fs");
const path = require("path");
const CommandsHelper = require("../../src/utils/CommandsHelper");

function extractCommandsFromCode(content, fileName) {
	const cmds = [];
	const cmdRegex = /new\s+Command\(\s*\{([\s\S]*?)\}\s*\)/g;
	let match;
	while ((match = cmdRegex.exec(content)) !== null) {
		const block = match[1];
		const nameMatch = block.match(/name:\s*["']([^"']+)["']/);
		const descMatch = block.match(/description:\s*["']([^"']+)["']/);
		const usageMatch = block.match(/usage:\s*["']([^"']+)["']/);
		const catMatch = block.match(/category:\s*["']([^"']+)["']/);

		if (nameMatch) {
			const name = nameMatch[1].trim();
			const desc = descMatch ? descMatch[1].trim() : "";
			const usage = usageMatch ? [usageMatch[1].trim()] : [`!${name}`];
			const category = catMatch ? catMatch[1].trim() : "jogos";

			cmds.push({
				cmd: `!${name}`,
				name: name.toLowerCase(),
				desc,
				category,
				usage,
				module: fileName,
				tags: [name, category]
			});
		}
	}
	return cmds;
}

async function exportCatalog() {
	console.log("[export_commands_catalog] Inicializando CommandsHelper...");
	const helper = CommandsHelper.getInstance();
	const outputPath = path.join(__dirname, "commands_catalog.json");

	const catalog = [];
	const seenCmds = new Set();

	// 1. Extrai do CommandsHelper (72 módulos)
	for (const h of helper.helpers) {
		const moduleTags = (h.tags || "")
			.split(",")
			.map((t) => t.trim().toLowerCase())
			.filter(Boolean);

		for (const c of h.cmds) {
			const rawCmd = (c.cmd || "").trim();
			if (!rawCmd) continue;

			const normalizedName = rawCmd.replace(/^!(g-)?/, "").toLowerCase();
			if (seenCmds.has(normalizedName)) continue;
			seenCmds.add(normalizedName);

			catalog.push({
				cmd: rawCmd,
				name: normalizedName,
				desc: c.desc || "",
				category: c.category || "geral",
				usage: Array.isArray(c.usage) ? c.usage : [],
				module: h.file,
				tags: moduleTags
			});
		}
	}

	// 2. Extrai de arquivos sem helper (ex: FazendaCommands, AFKCommands, CanaisCommands, EventCommands)
	const functionsPath = path.join(__dirname, "../../src/functions");
	const files = fs.readdirSync(functionsPath).filter((f) => f.endsWith(".js"));

	for (const file of files) {
		const filePath = path.join(functionsPath, file);
		const content = fs.readFileSync(filePath, "utf8");
		if (!/const\s+helper\s*=/.test(content) && /const\s+commands\s*=/.test(content)) {
			const extraCmds = extractCommandsFromCode(content, file);
			for (const ec of extraCmds) {
				if (!seenCmds.has(ec.name)) {
					seenCmds.add(ec.name);
					catalog.push(ec);
				}
			}
		}
	}

	fs.writeFileSync(outputPath, JSON.stringify(catalog, null, 2), "utf8");
	console.log(
		`[export_commands_catalog] Sucesso! Exportados ${catalog.length} comandos únicos para ${outputPath}`
	);
}

exportCatalog()
	.then(() => process.exit(0))
	.catch((err) => {
		console.error("[export_commands_catalog] Erro:", err);
		process.exit(1);
	});
