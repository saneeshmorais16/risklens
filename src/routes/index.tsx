import { createFileRoute } from "@tanstack/react-router";
import { RiskConsole } from "@/components/risk-console";

export const Route = createFileRoute("/")({
  component: Home,
});

function Home() {
  return <RiskConsole />;
}
