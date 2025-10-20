Energy Synergy Project

1. Project Goal

The primary goal of this project is to create a platform that analyzes and regroups the energy production and consumption data of multiple companies. By identifying companies that overproduce energy and those that need it, the platform aims to facilitate a local, win-win energy market. This allows producers to sell their surplus at a fair price and consumers to buy energy more affordably.

2. Core Concept

Many companies with renewable energy sources (like solar panels or wind turbines) often produce more than they consume, while others have high energy demands. This platform aggregates data from various sources to create a holistic view of a "Project Group," which is a collection of participating companies. The core analysis reveals periods of energy surplus and deficit within the group, highlighting opportunities for direct energy exchange.

3. Phase 1: Features

Data Ingestion & Management

The platform is designed to be flexible, accepting data in two primary formats:

Detailed Interval Data (CSV Upload):

Companies can upload a CSV file containing their production or consumption data at 15-minute intervals over a 24-hour period.

A company is either a producer or a consumer for a given dataset.

A downloadable CSV template is provided to ensure data is formatted correctly.

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

Analytics & Visualization

The main output of Phase 1 is a comprehensive analytics dashboard for each Project Group, featuring:

Aggregated View: Graphs showing the total combined production versus the total combined consumption of the group throughout the day.

Individual Contribution: Visualizations detailing the energy usage or production of each individual company in the group.

Surplus/Deficit Analysis: Clear identification of time periods where the group is self-sustaining, has excess energy to sell, or needs to buy from the grid.

4. Project Development Roadmap

[Current] Phase 1: Data Ingestion & Core Analysis:

Implement CSV upload and Profile creation.

Develop the analytics dashboard with key visualizations.

Refine the UI/UX to feel like a professional tool.

Phase 2: Pricing & Matching (Future):

Introduce functionality to add energy price data.

Develop an algorithm to suggest optimal energy trades between companies within a group.

Phase 3: Automation & Real-Time Data (Future):

Explore integration with real-time data APIs.

Automate the process of identifying and notifying companies of potential energy-sharing opportunities.