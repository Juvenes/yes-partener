# Energy Synergy Project

## 1. Project Goal

The primary goal of this project is to create a platform that analyzes and regroups the energy production and consumption data of multiple companies. By identifying companies that overproduce energy and those that need it, the platform aims to facilitate a local, win-win energy market. This allows producers to sell their surplus at a fair price and consumers to buy energy more affordably.

## 2. Core Concept

Many companies with renewable energy sources (like solar panels or wind turbines) often produce more than they consume, while others have high energy demands. This platform aggregates data from various sources to create a holistic view of a "Project Group," which is a collection of participating companies. The core analysis reveals periods of energy surplus and deficit within the group, highlighting opportunities for direct energy exchange.

## 3. Phase 1: Features

Data Ingestion & Management

The platform is designed to be flexible, accepting data in two primary formats:

Detailed Interval Data (CSV Upload):

Companies can upload a CSV file containing their production or consumption data at 15-minute intervals over a 24-hour period.


A downloadable CSV template is provided to ensure data is formatted correctly. That is the energy usage by a compagny (have 0 in production if don't generate or for who that generate sometime have  Consommation 0 and X production meaning over producted, )

CSV Template Format:

Time,Production,Consommation
00:00,0.15,0
00:15,0.12,0
...
08:00,0,0.66
08:15,0,0.71


Annual Data with Profiles:

For companies that only provide an annual production or consumption total, the platform uses a "Profile" system.

A Profile is a standardized 24-hour energy curve (e.g., "Office," "Factory") scaled to 1 kWh.

The platform multiplies the company's annual total by the selected profile to generate estimated interval data.

Users can create and manage these profiles on a dedicated page.
