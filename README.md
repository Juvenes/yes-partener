# Energy Synergy Project

## 1. Project Goal

The primary goal of this project is to create a platform that analyzes and regroups the energy production and consumption data of multiple companies. By identifying companies that overproduce energy and those that need it, the platform aims to facilitate a local, win-win energy market. This allows producers to sell their surplus at a fair price and consumers to buy energy more affordably.

## 2. Core Concept

Many companies with renewable energy sources (like solar panels or wind turbines) often produce more than they consume, while others have high energy demands. This platform aggregates data from various sources to create a holistic view of a "Project Group," which is a collection of participating companies. The core analysis reveals periods of energy surplus and deficit within the group, highlighting opportunities for direct energy exchange.

## 3. Phase 1: Features

### Unified data ingestion

All interval datasets are uploaded from a single page. Each CSV/Excel file is parsed, normalized on a common calendar (month, week, weekday, quarter-hour index) and stored as a downloadable Excel export. Tags replace ad‑hoc labels so datasets can be searched and filtered easily.

### Template

A downloadable CSV template is provided to ensure formatting is correct. Headers are:

```
Date+Quart time,consumption,injection
```

Values are expressed in kWh on a 15-minute step. Injection can be left blank when not applicable.
