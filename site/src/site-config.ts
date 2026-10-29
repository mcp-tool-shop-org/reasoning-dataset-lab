import type { SiteConfig } from '@mcptoolshop/site-theme';

export const config: SiteConfig = {
  title: 'reasoning-dataset-lab',
  description:
    'A map-first lab for datasets of puzzles, quizzes and verification items, where code proves the answer key before any model words the item.',
  logoBadge: 'RD',
  brandName: 'reasoning-dataset-lab',
  repoUrl: 'https://github.com/mcp-tool-shop-org/reasoning-dataset-lab',
  footerText:
    'MIT Licensed — built by <a href="https://github.com/mcp-tool-shop-org" style="color:var(--color-muted);text-decoration:underline">mcp-tool-shop-org</a>',

  hero: {
    badge: 'Alpha · Python 3.13+ · MIT',
    headline: 'The key comes from code,',
    headlineAccent: 'never from a judge.',
    description:
      'rdlab plans a dataset as a pre-registered map, builds items whose answer keys are constructed and proved by code, asks a local model only to put them into words, and records the provenance of every item.',
    primaryCta: { href: 'https://pypi.org/project/reasoning-dataset-lab/', label: 'Install from PyPI' },
    secondaryCta: { href: 'handbook/', label: 'Read the Handbook' },
    previews: [
      {
        label: 'Install',
        code: 'pip install reasoning-dataset-lab\n\n# Python 3.13 or newer.',
      },
      {
        label: 'Check a map',
        code: 'rdlab map check examples/t10-pilot-a.map.yaml\n\n# validates the cells, capacities and share rules,\n# then prints the map\'s canonical SHA-256.\n# no model, no network.',
      },
      {
        label: 'Fingerprint',
        code: 'rdlab project fingerprint . --map examples/t10-pilot-a.map.yaml\n\n# one hash for the five project files,\n# the map and canon/.',
      },
    ],
  },

  sections: [
    {
      kind: 'features',
      id: 'principles',
      title: 'How it keeps the key honest',
      subtitle: 'A key that a model chose is an opinion. A key that code built and proved is a fact about the item.',
      features: [
        {
          title: 'Prove first, word second',
          desc: 'Code builds the item from a hidden formal state and proves the key. A model is asked only for the wording, and blind readers from different families check it.',
        },
        {
          title: 'Pre-registered cells',
          desc: 'map.yaml declares every cell, its target and the generator\'s capacity before any item exists. A cell the generator cannot build stays in the map with capacity 0 and a reason.',
        },
        {
          title: 'One fingerprint',
          desc: 'Five project files, the map and the canon hash to one SHA-256, so a build says exactly what it was built from.',
        },
        {
          title: 'Certified value sets',
          desc: 'The values a claim is swapped within are certified before use. In the v3 run 44 of 50 sets certified, against 26 before repair. It is evidence, not proof, and the page says so.',
        },
        {
          title: 'Local and bounded',
          desc: 'Models run through Ollama on loopback. Anything else is refused. rdlab never pulls a model and never calls a cloud endpoint.',
        },
        {
          title: 'Not training data yet',
          desc: 'Nothing rdlab writes is training data until a person has checked it.',
        },
      ],
    },
    {
      kind: 'code-cards',
      id: 'usage',
      title: 'Use it',
      cards: [
        { title: 'Install', code: 'pip install reasoning-dataset-lab' },
        {
          title: 'Check a map, then fingerprint the project',
          code: 'rdlab map check examples/t10-pilot-a.map.yaml\nrdlab project fingerprint . --map examples/t10-pilot-a.map.yaml',
        },
      ],
    },
  ],
};
