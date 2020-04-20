import React, { useState, useEffect } from 'react';
import styled from 'styled-components';
import HideIcon from '../assets/show.svg';

const Container = styled.div`
    width: 400px;
    height: 100%;
    padding: 20px;
    padding-left: 50px;
    position: relative;
    background-color: #000;
    color: #fff;
    transition: 300ms;
`;

const Title = styled.span`
    display: inline-block;
    font-size: 24px;
    font-weight: 700;
    margin-bottom: 20px;
`;

const Hide = styled.div`
    width: 20px;
    height: 20px;
    display: flex;
    align-items: center;
    justify-content: center;
    position: absolute;
    top: 25px;
    left: 15px;
    cursor: pointer;
    background-color: #fff;

    &:hover > svg {
        transform: rotate(0);
    }

    & > svg {
        width: 10px;
        height: 10px;
        fill: #000;
        transform-origin: center;
        transform: rotate(180deg);
        transition: transform 250ms;
    }
`;

interface IDataPractice {
    name: string;
    about: string;
    segments: string[];
}

type Props = {
    dataPractice: IDataPractice;
    onHide: Function;
};

const initialStyle: React.CSSProperties = {
    opacity: 0,
    transform: 'translateX(50%)',
};

const Viewer = (props: Props) => {
    const [style, setStyle] = useState(initialStyle);

    // useEffect() dependency list is empty => run once
    useEffect(() => {
        setStyle({
            opacity: 1,
        });
    }, []);

    return (
        <Container style={style}>
            <Title>{props.dataPractice.name}</Title>

            <Hide
                onClick={(_) => {
                    setStyle(initialStyle);

                    // hides viewer after 200ms delay
                    props.onHide();
                }}>
                <HideIcon />
            </Hide>

            <ul>
                {props.dataPractice.segments.map((s) => (
                    <li style={{ marginBottom: 20 }}>
                        <span>{s}</span>
                    </li>
                ))}
            </ul>

            {props.dataPractice.segments.length === 0 && <span>No segments found.</span>}
        </Container>
    );
};

export default Viewer;
